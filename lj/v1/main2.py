"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import torch
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import json

from models.LAIN2 import build_detector
from utils.args2 import get_args

from engine2 import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

warnings.filterwarnings("ignore")


def main(rank, args):
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=rank
        )
    except Exception as e:
        print(f"[ERROR] Rank {rank}: 分布式初始化失败 → {e}")
        raise

    # Fix seed（增强随机性，避免训练固化）
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(rank)

    # 解析CLIP模型名称（原版逻辑+适配）
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'

    # ========== 新增：打印关键参数（调试用） ==========
    if rank == 0:
        print(f"\n[核心参数]")
        print(f"  - 数据集: {args.dataset} | 零样本类型: {args.zs_type if args.zs else 'None'}")
        print(f"  - 适配器模式: {args.adapter_mode if args.use_text_adapter else '关闭'} | LoRA Rank: {args.lora_rank}")
        print(f"  - 训练轮数: {args.epochs} | 批次大小: {args.batch_size}")
        print(f"  - 适配器学习率: {args.lr_adapter_complex} | ViT学习率: {args.lr_vit}")

    # 加载数据集（原版逻辑+增强日志）
    if rank == 0:
        print(f"\n[加载数据集] 训练集: {args.partitions[0]} | 测试集: {args.partitions[1]}")
    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
                           clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
                           num_classes=args.num_classes, args=args)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
                          clip_model_name=args.clip_model_name, args=args)

    # 构建数据加载器（原版逻辑）
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True  # 显式开启shuffle，避免训练数据重复
    )
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,  # 开启pin_memory加速
        sampler=train_sampler
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=test_sampler
    )

    # 模型初始化配置（原版逻辑+调试日志）
    args.human_idx = 0
    object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction
    object_to_target = trainset.dataset.object_class_to_target_class

    if rank == 0:
        print(f'[INFO]: num_classes = {args.num_classes} | 适配器启用状态 = {args.use_text_adapter}')

    # ========== 新增：模型构建前校验参数 ==========
    if not hasattr(args, 'use_text_adapter'):
        args.use_text_adapter = False
    if rank == 0 and not args.use_text_adapter:
        print(f"[WARNING] 未开启--use_text_adapter！适配器参数将不会被加载")

    lain = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction,
                          clip_model_path=args.clip_dir_vit)

    # 模型移至GPU（新增，避免后续设备不匹配）
    lain = lain.cuda(rank)

    # 适配HICO-DET目标类别映射（原版逻辑）
    if args.dataset == 'hicodet' and args.eval:
        lain.object_class_to_target_class = testset.dataset.object_class_to_target_class

    # 加载预训练模型（原版逻辑+增强校验）
    scaler = None
    start_epoch = 0
    start_iteration = 0
    if os.path.exists(args.resume):
        if rank == 0:
            print(f"===>>> Rank {rank}: 从 checkpoint {args.resume} 恢复训练")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')  # 直接加载到对应GPU
        # 宽松加载模型参数，避免适配器参数不匹配导致加载失败
        missing_keys, unexpected_keys = lain.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if rank == 0:
            print(f"[模型加载] 缺失参数: {len(missing_keys)} | 多余参数: {len(unexpected_keys)}")
            if len(missing_keys) > 0 and 'text_adapter' in str(missing_keys):
                print(f"[WARNING] 适配器参数缺失 → {missing_keys}")
        start_epoch = checkpoint.get('epoch', 0)
        start_iteration = checkpoint.get('iteration', 0)
        if 'scaler_state_dict' in checkpoint:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        if rank == 0:
            print(f"=> Rank {rank}: 从随机初始化开始训练")

    # 初始化训练引擎（原版逻辑+参数增强）
    engine = CustomisedDLE(
        lain, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,  # 必须开启，适配适配器参数
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args
    )

    # 缓存模式（原版逻辑）
    if args.cache:
        if rank == 0:
            print(f"\n[缓存模式] 数据集: {args.dataset} | 缓存目录: {args.output_dir}")
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    # 评估模式（原版逻辑）
    if args.eval:
        if rank == 0:
            print(f"\n[评估模式] 开始评估 {args.dataset} 数据集...")
        lain.eval()
        if args.dataset == 'vcoco':
            import eval_vcoco
            ret = engine.cache_vcoco(test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file = 'vcoco/data/splits/vcoco_test.ids'
            vcocoeval = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
            det_file = 'vcoco_cache/cache.pkl'
            ap = vcocoeval._do_eval(ret, ovr_thresh=0.5)
            if rank == 0:
                print(f"V-COCO AP: {ap}")
            return
        ap = engine.test_hico(test_loader, args)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        if rank == 0:
            print(
                f"The mAP is {ap.mean() * 100:.2f},"
                f" rare: {ap[rare].mean() * 100:.2f},"
                f" none-rare: {ap[non_rare].mean() * 100:.2f},"
            )
            if args.zs:
                zs_hoi_idx = hico_unseen_index[args.zs_type]
                print(f'>>> zero-shot setting({args.zs_type}!!)')
                ap_unseen = []
                ap_seen = []
                for i, value in enumerate(ap):
                    if i in zs_hoi_idx:
                        ap_unseen.append(value)
                    else:
                        ap_seen.append(value)

                ap_unseen = torch.as_tensor(ap_unseen).mean()
                ap_seen = torch.as_tensor(ap_seen).mean()
                print(
                    f"full mAP: {ap.mean() * 100:.2f}",
                    f"unseen: {ap_unseen * 100:.2f}",
                    f"seen: {ap_seen * 100:.2f}",
                )
        return

    # ========== 核心修复：参数冻结/解冻逻辑 ==========
    if rank == 0:
        print(f"\n[参数配置] 开始冻结主干 + 解冻适配器...")

    # 1. 冻结detector主干（保留+显式日志）
    for p in lain.detector.parameters():
        p.requires_grad = False
    if rank == 0:
        print(f"  ✓ 冻结Detector主干参数")

    # 2. 冻结CLIP Head非核心层（保留+精细化控制）
    clip_trainable = []
    clip_frozen = []
    for n, p in lain.clip_head.named_parameters():
        if (n.startswith('visual.positional_embedding') or
                n.startswith('visual.ln_post') or
                n.startswith('visual.proj') or
                'adaptermlp' in n or
                "prompt_learner" in n or
                'visual_prompt' in n):
            p.requires_grad = True
            clip_trainable.append(n)
        else:
            p.requires_grad = False
            clip_frozen.append(n)
    if rank == 0:
        print(f"  ✓ CLIP Head可训练参数: {len(clip_trainable)}个 | 冻结参数: {len(clip_frozen)}个")

    # 3. 强制解冻所有适配器参数（核心修复+全量校验）
    adapter_unfreeze_count = 0
    adapter_params = []
    for n, p in lain.named_parameters():
        if 'text_adapter' in n:
            p.requires_grad = True
            adapter_unfreeze_count += 1
            adapter_params.append(n)
            if rank == 0:
                print(f"  ✓ 解冻适配器参数: {n} | 梯度状态: {p.requires_grad}")

    # 强制检查：确保适配器参数被解冻
    if rank == 0:
        if adapter_unfreeze_count == 0:
            if args.use_text_adapter:
                raise ValueError("ERROR: 未找到任何text_adapter参数！请检查模型构建逻辑")
            else:
                print(f"[WARNING] 未开启use_text_adapter，跳过适配器参数解冻")
        else:
            print(f"\n[INFO] 成功解冻{adapter_unfreeze_count}个适配器参数")

    # 4. 优化器分组（修复学习率+显式参数统计）
    clip_head_params = [p for n, p in lain.clip_head.named_parameters() if p.requires_grad]
    adapter_params = [p for n, p in lain.named_parameters() if p.requires_grad and 'text_adapter' in n]
    other_params = [p for n, p in lain.named_parameters() if
                    p.requires_grad and 'clip_head' not in n and 'text_adapter' not in n]

    param_dicts = []
    if clip_head_params:
        param_dicts.append({
            "params": clip_head_params,
            "lr": args.lr_vit,  # 使用指令传入的lr_vit
            "weight_decay": args.weight_decay
        })
    if adapter_params:
        param_dicts.append({
            "params": adapter_params,
            "lr": args.lr_adapter_complex,  # 使用指令传入的适配器学习率
            "weight_decay": 0.0  # 适配器关闭权重衰减，避免梯度消失
        })
    if other_params:
        param_dicts.append({
            "params": other_params,
            "lr": args.lr_head,
            "weight_decay": args.weight_decay
        })

    # 打印优化器分组信息
    if rank == 0:
        print(f"\n[优化器配置]")
        print(f"  - CLIP Head组: {len(clip_head_params)}个参数 | LR: {args.lr_vit}")
        print(f"  - 适配器组: {len(adapter_params)}个参数 | LR: {args.lr_adapter_complex}")
        print(f"  - 其他参数组: {len(other_params)}个参数 | LR: {args.lr_head}")

    # 初始化优化器（使用指令传入的参数）
    optim = torch.optim.AdamW(
        param_dicts,
        lr=args.lr_vit,  # 基础LR
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)  # 显式设置betas，增强稳定性
    )

    # 学习率调度器（适配30epoch+余弦退火）
    if args.epochs == 30 and not args.resume:
        # 余弦退火更适合短周期训练，避免StepLR学习率突变
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=args.epochs,
            eta_min=1e-6,  # 最小学习率
            last_epoch=-1
        )
        if rank == 0:
            print(f"  ✓ 使用余弦退火调度器 (T_max={args.epochs})")
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=args.lr_drop,
            gamma=0.1  # 学习率衰减系数
        )
        if rank == 0:
            print(f"  ✓ 使用StepLR调度器 (step_size={args.lr_drop})")

    # ========== 恢复训练状态（原版逻辑+增强） ==========
    if args.resume:
        # 可选：加载优化器和调度器状态
        if 'optim_state_dict' in checkpoint:
            optim.load_state_dict(checkpoint['optim_state_dict'])
            if rank == 0:
                print(f"  ✓ 恢复优化器状态")
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if rank == 0:
                print(f"  ✓ 恢复学习率调度器状态")
        engine.update_state_key(
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            epoch=start_epoch,
            iteration=start_iteration,
            scaler=scaler
        )
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    # 保存参数配置（原版逻辑+增强）
    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        args_dict = vars(args)
        # 过滤敏感/冗余参数
        save_dict = {k: v for k, v in args_dict.items() if not isinstance(v, (torch.Tensor, np.ndarray))}
        with open(os.path.join(args.output_dir, 'train_config.json'), 'w') as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        print(f"\n[训练准备完成] 配置文件已保存至: {os.path.join(args.output_dir, 'train_config.json')}")

    # 启动训练（核心逻辑）
    if rank == 0:
        print(f"\n========== 开始训练 ({args.epochs} epochs) ==========\n")
    engine(args.epochs)


if __name__ == '__main__':
    args = get_args()
    print(f"[启动参数] {args}")

    # 环境配置（增强稳定性）
    if args.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ["MASTER_PORT"] = args.port
        print(f"[调试模式] WANDB禁用 | MASTER_PORT={args.port}")
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # 开启分布式调试日志

    print(f'WORLD_SIZE = {os.environ.get("WORLD_SIZE", 1)}')

    os.environ["MASTER_ADDR"] = "localhost"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank = local_rank

    # 初始化WandB（仅主进程）
    if local_rank == 0 and not args.debug:
        wandb.init(
            project='LAIN-Adapter',
            name=args.output_dir.split('/')[-1],
            config=vars(args)
        )
        print(f"[WandB] 初始化完成 | Project: LAIN-Adapter | Name: {args.output_dir.split('/')[-1]}")

    # 设置分布式参数（增强鲁棒性）
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.world_size > 1 and local_rank == 0:
        print(f"[分布式训练] 进程数: {args.world_size} | 本地Rank: {local_rank}")

    # 启动主函数（异常捕获）
    try:
        main(local_rank, args)
    except Exception as e:
        print(f"[ERROR] 训练失败 → {e}")
        raise
    finally:
        # 清理分布式进程
        dist.destroy_process_group()
        if local_rank == 0 and not args.debug:
            wandb.finish()
        print(f"\n[训练结束] Rank {local_rank} 进程已清理")