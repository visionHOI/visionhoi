"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.
Integrated version: supports SceneGate + TextSemanticAdapter + Robust training.
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

# 使用整合后的 LAIN 和 args
from models.LAIN import build_detector
from utils.args import get_args

# 注意：这里使用 engine1（增强版 engine），若实际模块名不同请调整
from engine import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

warnings.filterwarnings("ignore")


def main(rank, args):
    # -------------------- 分布式初始化 --------------------
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

    # -------------------- 随机种子（增强可复现性） --------------------
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(rank)

    # -------------------- 解析 CLIP 模型名称 --------------------
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'

    # -------------------- 打印关键参数（仅主进程） --------------------
    if rank == 0:
        print(f"\n[核心参数]")
        print(f"  - 数据集: {args.dataset} | 零样本类型: {args.zs_type if args.zs else 'None'}")
        print(f"  - 文本适配器: {args.use_text_adapter} | 语义适配器: {args.use_semantic_adapter}")
        print(f"  - SceneGate: {args.use_scene_gate} (类型: {args.scene_gate_type})")
        print(f"  - 训练轮数: {args.epochs} | 批次大小: {args.batch_size}")

    # -------------------- 数据集加载 --------------------
    if rank == 0:
        print(f"\n[加载数据集] 训练集: {args.partitions[0]} | 测试集: {args.partitions[1]}")
    trainset = DataFactory(
        name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
        clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
        num_classes=args.num_classes, args=args
    )
    testset = DataFactory(
        name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
        clip_model_name=args.clip_model_name, args=args
    )

    # -------------------- 数据加载器 --------------------
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
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

    # -------------------- 模型构建 --------------------
    args.human_idx = 0
    object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction
    object_to_target = trainset.dataset.object_class_to_target_class

    if rank == 0:
        print(f'[INFO]: num_classes = {args.num_classes} | 文本适配器启用 = {args.use_text_adapter}')
        print(f'[INFO]: use_scene_gate = {args.use_scene_gate}, scene_gate_type = {args.scene_gate_type}')

    lain = build_detector(
        args, object_to_target,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        clip_model_path=args.clip_dir_vit
    )
    lain = lain.cuda(rank)

    # 适配 HICO-DET 目标类别映射（评估时）
    if args.dataset == 'hicodet' and args.eval:
        lain.object_class_to_target_class = testset.dataset.object_class_to_target_class

    # -------------------- 加载 checkpoint --------------------
    scaler = None
    start_epoch = 0
    start_iteration = 0
    if os.path.exists(args.resume):
        if rank == 0:
            print(f"===>>> Rank {rank}: 从 checkpoint {args.resume} 恢复训练")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')
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

    # -------------------- 初始化训练引擎 --------------------
    engine = CustomisedDLE(
        lain, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,  # 适配器参数可能不全部使用
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args
    )

    # -------------------- 缓存模式 --------------------
    if args.cache:
        if rank == 0:
            print(f"\n[缓存模式] 数据集: {args.dataset} | 缓存目录: {args.output_dir}")
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    # -------------------- 评估模式 --------------------
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
            ap = vcocoeval._do_eval(ret, ovr_thresh=0.5)
            if rank == 0:
                print(f"V-COCO AP: {ap}")
            return
        ap = engine.test_hico(test_loader, args)
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

    # -------------------- 训练模式：参数冻结/解冻 --------------------
    if rank == 0:
        print(f"\n[参数配置] 开始冻结主干 + 解冻可训练组件...")

    # 1. 冻结 DETR 主干
    for p in lain.detector.parameters():
        p.requires_grad = False
    if rank == 0:
        print(f"  ✓ 冻结 Detector 主干参数")

    # 2. 冻结 CLIP Head 非核心层，解冻指定层
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
        print(f"  ✓ CLIP Head 可训练参数: {len(clip_trainable)}个 | 冻结: {len(clip_frozen)}个")

    # 3. 解冻文本适配器参数（如果启用）
    if args.use_text_adapter:
        adapter_unfreeze_count = 0
        for n, p in lain.named_parameters():
            if 'text_adapter' in n:
                p.requires_grad = True
                adapter_unfreeze_count += 1
        if rank == 0:
            print(f"  ✓ 解冻文本适配器参数: {adapter_unfreeze_count}个")
    else:
        if rank == 0:
            print(f"  - 文本适配器未启用，跳过解冻")

    # 4. 解冻 SceneGate 参数（如果启用）
    if args.use_scene_gate:
        scene_gate_count = 0
        for n, p in lain.named_parameters():
            if 'scene_gate' in n:
                p.requires_grad = True
                scene_gate_count += 1
        if rank == 0:
            print(f"  ✓ 解冻 SceneGate 参数: {scene_gate_count}个")
    else:
        if rank == 0:
            print(f"  - SceneGate 未启用，跳过解冻")

    # 5. 解冻其他可能需要的参数（如 priors_downproj, query_proj 等）
    #    这些参数通常需要训练，但已在 LAIN 初始化时默认 requires_grad=True，无需额外操作。
    #    若需控制，可在此添加。

    # -------------------- 优化器分组 --------------------
    clip_head_params = [p for n, p in lain.clip_head.named_parameters() if p.requires_grad]
    # 适配器参数（文本适配器 + SceneGate 可能已包含在 other_params 中，但为清晰可单独分组）
    adapter_params = [p for n, p in lain.named_parameters() if p.requires_grad and 'text_adapter' in n]
    scene_gate_params = [p for n, p in lain.named_parameters() if p.requires_grad and 'scene_gate' in n]
    other_params = [p for n, p in lain.named_parameters() if
                    p.requires_grad and 'clip_head' not in n and 'text_adapter' not in n and 'scene_gate' not in n]

    param_dicts = []
    if clip_head_params:
        param_dicts.append({
            "params": clip_head_params,
            "lr": args.lr_vit,
            "weight_decay": args.weight_decay
        })
    if adapter_params:
        # 若 args 中未定义 lr_adapter_complex，使用 args.lr_vit 或 args.lr_head
        adapter_lr = getattr(args, 'lr_adapter_complex', args.lr_head)
        param_dicts.append({
            "params": adapter_params,
            "lr": adapter_lr,
            "weight_decay": 0.0  # 适配器通常不加 weight decay
        })
    if scene_gate_params:
        param_dicts.append({
            "params": scene_gate_params,
            "lr": args.lr_head,   # 使用头部学习率
            "weight_decay": args.weight_decay
        })
    if other_params:
        param_dicts.append({
            "params": other_params,
            "lr": args.lr_head,
            "weight_decay": args.weight_decay
        })

    if rank == 0:
        print(f"\n[优化器配置]")
        print(f"  - CLIP Head 组: {len(clip_head_params)} 参数 | LR: {args.lr_vit}")
        print(f"  - 文本适配器组: {len(adapter_params)} 参数 | LR: {getattr(args, 'lr_adapter_complex', args.lr_head)}")
        print(f"  - SceneGate 组: {len(scene_gate_params)} 参数 | LR: {args.lr_head}")
        print(f"  - 其他参数组: {len(other_params)} 参数 | LR: {args.lr_head}")

    optim = torch.optim.AdamW(
        param_dicts,
        lr=args.lr_vit,  # 基础 LR，实际被分组覆盖
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # -------------------- 学习率调度器 --------------------
    if args.epochs == 30 and not args.resume:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=args.epochs,
            eta_min=1e-6,
            last_epoch=-1
        )
        if rank == 0:
            print(f"  ✓ 使用余弦退火调度器 (T_max={args.epochs})")
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=args.lr_drop,
            gamma=0.1
        )
        if rank == 0:
            print(f"  ✓ 使用 StepLR 调度器 (step_size={args.lr_drop})")

    # -------------------- 恢复训练状态 --------------------
    if args.resume:
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

    # -------------------- 保存配置 --------------------
    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        args_dict = vars(args)
        # 过滤不可序列化的值
        save_dict = {k: v for k, v in args_dict.items() if not isinstance(v, (torch.Tensor, np.ndarray))}
        with open(os.path.join(args.output_dir, 'train_config.json'), 'w') as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        print(f"\n[训练准备完成] 配置文件已保存至: {os.path.join(args.output_dir, 'train_config.json')}")

    # -------------------- 启动训练 --------------------
    if rank == 0:
        print(f"\n========== 开始训练 ({args.epochs} epochs) ==========\n")
    engine(args.epochs)


if __name__ == '__main__':
    args = get_args()
    print(f"[启动参数] {args}")

    # -------------------- 环境配置 --------------------
    if args.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ["MASTER_PORT"] = args.port
        print(f"[调试模式] WANDB 禁用 | MASTER_PORT={args.port}")
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # 开启分布式调试日志

    print(f'WORLD_SIZE = {os.environ.get("WORLD_SIZE", 1)}')

    os.environ["MASTER_ADDR"] = "localhost"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank = local_rank

    # -------------------- WandB 初始化（仅主进程） --------------------
    if local_rank == 0 and not args.debug:
        wandb.init(
            project='LAIN-Integrated',
            name=args.output_dir.split('/')[-1],
            config=vars(args)
        )
        print(f"[WandB] 初始化完成 | Project: LAIN-Integrated | Name: {args.output_dir.split('/')[-1]}")

    # -------------------- 分布式参数 --------------------
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.world_size > 1 and local_rank == 0:
        print(f"[分布式训练] 进程数: {args.world_size} | 本地 Rank: {local_rank}")

    # -------------------- 启动主函数 --------------------
    try:
        main(local_rank, args)
    except Exception as e:
        print(f"[ERROR] 训练失败 → {e}")
        raise
    finally:
        dist.destroy_process_group()
        if local_rank == 0 and not args.debug:
            wandb.finish()
        print(f"\n[训练结束] Rank {local_rank} 进程已清理")
