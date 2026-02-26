import os
import torch
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import wandb
import json
from datetime import datetime

from models.LAIN import build_detector
from utils.args import get_args
from engine import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

# 全局配置
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True  # 加速CUDA运算
torch.backends.cudnn.deterministic = False  # 非确定性算法加速


# ====================== 全局补丁：给RandomSampler添加set_epoch方法 ======================
def dummy_set_epoch(self, epoch):
    """空实现set_epoch，兼容分布式采样器逻辑"""
    pass


RandomSampler.set_epoch = dummy_set_epoch


def setup_distributed(args):
    """初始化分布式环境（鲁棒版）"""
    os.environ["MASTER_ADDR"] = args.master_addr if hasattr(args, 'master_addr') else "localhost"
    os.environ["MASTER_PORT"] = args.port if hasattr(args, 'port') else "29500"

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.local_rank
        )
    except Exception as e:
        print(f"[WARNING] 分布式初始化失败: {e}，使用单机单卡模拟")
        # 模拟分布式接口
        import types
        def mock_get_rank():
            return 0

        dist.get_rank = mock_get_rank
        dist.get_world_size = lambda: 1


def set_random_seed(seed, rank=0):
    """设置全局随机种子"""
    final_seed = seed + rank
    torch.manual_seed(final_seed)
    np.random.seed(final_seed)
    random.seed(final_seed)
    torch.cuda.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)
    print(f"[INFO] 已设置随机种子: {final_seed}")


def process_clip_model_name(args):
    """标准化CLIP模型名称"""
    if not hasattr(args, 'clip_dir_vit') or not args.clip_dir_vit:
        raise ValueError("必须指定CLIP模型路径: clip_dir_vit")

    clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    name_mapping = {
        'ViT-B-16': 'ViT-B/16',
        'ViT-L-14-336px': 'ViT-L/14@336px',
        'ViT-B-32': 'ViT-B/32',
        'ViT-L-14': 'ViT-L/14'
    }
    args.clip_model_name = name_mapping.get(clip_model_name, clip_model_name)
    print(f"[INFO] 标准化CLIP模型名称: {args.clip_model_name}")


def build_data_loaders(args):
    """构建训练/测试DataLoader（模块化）"""
    # 加载数据集
    print(f"[INFO] 加载数据集: {args.dataset} (训练集: {args.partitions[0]}, 测试集: {args.partitions[1]})")
    trainset = DataFactory(
        name=args.dataset,
        partition=args.partitions[0],
        data_root=args.data_root,
        clip_model_name=args.clip_model_name,
        zero_shot=args.zs if hasattr(args, 'zs') else False,
        zs_type=args.zs_type if hasattr(args, 'zs_type') else None,
        num_classes=args.num_classes,
        args=args
    )

    testset = DataFactory(
        name=args.dataset,
        partition=args.partitions[1],
        data_root=args.data_root,
        clip_model_name=args.clip_model_name,
        args=args
    )

    # 构建DataLoader
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,  # 开启pin_memory加速
        drop_last=True,
        shuffle=True,
        persistent_workers=True if args.num_workers > 0 else False  # 持久化worker加速
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # 数据集元信息
    args.human_idx = 0
    args.object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction
    args.object_to_target = trainset.dataset.object_class_to_target_class

    print(f'[INFO]: 类别数 num_classes = {args.num_classes}')
    print(f'[INFO]: 训练集样本数 = {len(trainset)}, 测试集样本数 = {len(testset)}')

    return train_loader, test_loader


def load_model(args, object_to_target, object_n_verb_to_interaction):
    """构建并加载模型"""
    # 构建模型
    lain = build_detector(
        args,
        object_to_target,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        clip_model_path=args.clip_dir_vit
    )

    # 适配HICO-DET评估
    if args.dataset == 'hicodet' and args.eval:
        lain.object_class_to_target_class = args.object_to_target_test

    # 加载权重
    if os.path.exists(args.resume):
        print(f"[INFO] 从 checkpoint 恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        # 兼容加载（忽略不匹配的参数）
        missing_keys, unexpected_keys = lain.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False
        )
        if missing_keys:
            print(f"[WARNING] 缺失参数: {missing_keys[:5]}")  # 只打印前5个
        if unexpected_keys:
            print(f"[WARNING] 多余参数: {unexpected_keys[:5]}")
        epoch_start = checkpoint.get('epoch', 0)
        iteration_start = checkpoint.get('iteration', 0)
    else:
        print("[INFO] 使用随机初始化模型开始训练")
        epoch_start = 0
        iteration_start = 0

    # 移动模型到GPU
    lain = lain.cuda()
    return lain, epoch_start, iteration_start


def freeze_unfreeze_params(lain, args):
    """冻结/解冻参数（精细化控制）"""
    # 冻结DETR主干
    for p in lain.detector.parameters():
        p.requires_grad = False
    print("[INFO] 已冻结DETR主干网络")

    # 解冻指定层
    unfrozen_params = []
    for n, p in lain.detector.named_parameters():
        if any(key in n for key in ['class_embed', 'bbox_embed', 'hoi_head']):
            p.requires_grad = True
            unfrozen_params.append(n)
            print(f"[INFO] 解冻 DETR 参数: {n}")

    # CLIP Head 解冻逻辑（可配置）
    clip_unfreeze_keys = ['text_adapter', 'A', 'B', 'adaptermlp', 'prompt_learner', 'visual_prompt']
    for n, p in lain.clip_head.named_parameters():
        if any(key in n for key in clip_unfreeze_keys):
            p.requires_grad = True
            unfrozen_params.append(n)
            print(f"[INFO] 解冻 CLIP Head 参数: {n}")
        else:
            p.requires_grad = False

    # 验证可训练参数
    trainable_params = sum(p.numel() for p in lain.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lain.parameters())
    print(f"[INFO] 可训练参数: {trainable_params / 1e6:.2f}M / 总参数: {total_params / 1e6:.2f}M")


def build_optimizer_scheduler(lain, args):
    """构建优化器和学习率调度器"""
    # 分学习率配置
    param_dicts = [
        {
            "params": [p for n, p in lain.clip_head.named_parameters() if p.requires_grad],
            "lr": args.lr_vit,
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in lain.named_parameters() if p.requires_grad and 'clip_head' not in n],
            "lr": args.lr_head,
            "weight_decay": args.weight_decay
        },
    ]

    # 构建优化器
    optim = torch.optim.AdamW(
        param_dicts,
        lr=args.lr_vit,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 构建学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=args.lr_drop,
        gamma=0.1
    )

    return optim, lr_scheduler


def save_args_config(args, output_dir):
    """保存参数配置（带时间戳）"""
    os.makedirs(output_dir, exist_ok=True)
    args_dict = args.__dict__.copy()
    args_dict['train_start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 保存为JSON
    with open(os.path.join(output_dir, 'args_config.json'), 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    # 保存为TXT（方便阅读）
    with open(os.path.join(output_dir, 'args_config.txt'), 'w', encoding='utf-8') as f:
        for k, v in args_dict.items():
            f.write(f"{k}: {v}\n")


def main(rank, args):
    """主训练函数"""
    # 1. 初始化分布式环境
    setup_distributed(args)

    # 2. 设置随机种子
    set_random_seed(args.seed, rank)

    # 3. 设置GPU
    torch.cuda.set_device(rank)
    print(f"[INFO] 使用GPU: {rank}")

    # 4. 处理CLIP模型名称
    process_clip_model_name(args)

    # 5. 构建数据加载器
    train_loader, test_loader = build_data_loaders(args)
    if args.dataset == 'hicodet' and args.eval:
        args.object_to_target_test = test_loader.dataset.object_class_to_target_class

    # 6. 构建并加载模型
    lain, epoch_start, iteration_start = load_model(
        args,
        args.object_to_target,
        args.object_n_verb_to_interaction
    )

    # 7. 初始化训练引擎
    engine = CustomisedDLE(
        lain,
        train_loader,
        max_norm=args.clip_max_norm if hasattr(args, 'clip_max_norm') else 1.0,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=False,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args
    )

    # 8. 缓存/评估逻辑
    if args.cache:
        print("[INFO] 执行缓存逻辑，退出训练")
        return
    if args.eval:
        print("[INFO] 执行评估逻辑，退出训练")
        engine.evaluate()  # 假设engine有evaluate方法
        return

    # 9. 冻结/解冻参数
    freeze_unfreeze_params(lain, args)

    # 10. 构建优化器和调度器
    optim, lr_scheduler = build_optimizer_scheduler(lain, args)

    # 11. 恢复训练状态
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    # 12. 更新引擎状态
    engine.update_state_key(
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        epoch=epoch_start,
        iteration=iteration_start,
        scaler=scaler
    )

    # 13. 保存参数配置
    save_args_config(args, args.output_dir)

    # 14. 启动训练
    print(f"[INFO] 开始训练，总轮数: {args.epochs}, 起始轮数: {epoch_start}")
    engine(args.epochs)

    # 15. 训练结束清理
    dist.destroy_process_group() if dist.is_initialized() else None
    print("[INFO] 训练完成")


if __name__ == '__main__':
    # 解析参数
    args = get_args()
    print("[INFO] 输入参数:")
    for k, v in sorted(args.__dict__.items()):
        print(f"  {k}: {v}")

    # 配置WandB
    os.environ['WANDB_MODE'] = 'offline'
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_DISABLED"] = "true" if args.disable_wandb else "false"  # 新增禁用参数

    # 单卡配置
    args.local_rank = 0
    args.world_size = 1
    args.master_addr = "localhost"

    # 初始化WandB（容错）
    try:
        wandb.init(
            project='LAIN',
            name=args.output_dir,
            config=args.__dict__
        )
    except Exception as e:
        print(f"[WARNING] WandB初始化失败: {e}")

    # 运行主函数
    main(args.local_rank, args)