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

from models.LAIN_v2 import build_detector
from utils.args_v2 import get_args

from engine_v2 import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

warnings.filterwarnings("ignore")


def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(rank)

    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
                           clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
                           num_classes=args.num_classes, args=args)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
                          clip_model_name=args.clip_model_name, args=args)

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        sampler=DistributedSampler(
            trainset,
            num_replicas=args.world_size,
            rank=rank
        )
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            testset, shuffle=False, drop_last=False
        )
    )

    args.human_idx = 0
    object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction
    object_to_target = trainset.dataset.object_class_to_target_class

    print('[INFO]: num_classes', args.num_classes)
    lain = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction,
                          clip_model_path=args.clip_dir_vit)

    if args.dataset == 'hicodet' and args.eval:  ## after building model, manually change obj_to_target
        lain.object_class_to_target_class = testset.dataset.object_class_to_target_class

    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        lain.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    engine = CustomisedDLE(
        lain, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
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
            print(ap)
            return
        ap = engine.test_hico(test_loader, args)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
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

    # ====================== 核心调整1：冻结DETR检测器 ======================
    for p in lain.detector.parameters():
        p.requires_grad = False

    # ====================== 核心调整2：CLIP参数冻结逻辑（适配文本适配器） ======================
    # 新增：打印参数冻结/可训练状态（仅rank 0）
    if rank == 0:
        print("\n=== 参数可训练状态 ===")
        trainable_params = []
        frozen_params = []

    for n, p in lain.clip_head.named_parameters():
        # 原有可训练参数逻辑
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj'):
            p.requires_grad = True
            if rank == 0:
                trainable_params.append(n)
        elif 'adaptermlp' in n or "prompt_learner" in n:
            p.requires_grad = True
            if rank == 0:
                trainable_params.append(n)
        elif 'visual_prompt' in n:
            p.requires_grad = True
            if rank == 0:
                trainable_params.append(n)
        # 新增：文本适配器参数强制可训练（仅当启用适配器时）
        elif args.use_text_adapter and 'text_adapter' in n:
            p.requires_grad = True
            if rank == 0:
                trainable_params.append(n)
        # 其他参数冻结（包括CLIP文本编码器）
        else:
            p.requires_grad = False
            if rank == 0:
                frozen_params.append(n)

    # 打印参数状态（仅rank 0）
    if rank == 0:
        print(f"可训练参数数量: {len(trainable_params)}")
        for param_name in trainable_params[:10]:  # 仅打印前10个
            print(f"  - {param_name}")
        if len(trainable_params) > 10:
            print(f"  - ... 共{len(trainable_params)}个可训练参数")
        print(f"\n冻结参数数量: {len(frozen_params)}")

    # ====================== 核心调整3：构建参数组（单独配置适配器学习率） ======================
    # 初始化参数组
    param_dicts = []

    # 1. 提取CLIP原有可训练参数（排除适配器）
    clip_params = []
    for n, p in lain.clip_head.named_parameters():
        if p.requires_grad and not (args.use_text_adapter and 'text_adapter' in n):
            clip_params.append(p)

    # 2. 提取文本适配器参数（仅当启用时）
    adapter_params = []
    if args.use_text_adapter:
        adapter_params = [
            p for n, p in lain.named_parameters()
            if 'text_adapter' in n and p.requires_grad
        ]
        if rank == 0:
            print(f"\n[适配器配置] 可训练参数数量: {sum(p.numel() for p in adapter_params) if adapter_params else 0}")

    # 3. 提取其他可训练参数（非CLIP）
    other_params = []
    for n, p in lain.named_parameters():
        if p.requires_grad and 'clip_head' not in n and not (args.use_text_adapter and 'text_adapter' in n):
            other_params.append(p)

    # 构建参数组（按学习率分组）
    # 组1：CLIP原有可训练参数（使用lr_vit）
    if clip_params:
        param_dicts.append({
            "params": clip_params,
            "lr": args.lr_vit
        })

    # 组2：文本适配器参数（单独学习率，默认5e-4）
    if adapter_params:
        # 适配args：如果没有lr_text_adapter参数，设置默认值
        lr_adapter = args.lr_text_adapter if hasattr(args, 'lr_text_adapter') else 5e-4
        param_dicts.append({
            "params": adapter_params,
            "lr": lr_adapter,
            "weight_decay": 0.0  # 适配器参数通常禁用权重衰减
        })

    # 组3：其他参数（使用lr_head）
    if other_params:
        param_dicts.append({
            "params": other_params,
            "lr": args.lr_head if hasattr(args, 'lr_head') else 1e-4
        })

    # 打印参数组配置（仅rank 0）
    if rank == 0:
        print("\n=== 优化器参数组配置 ===")
        total_trainable_params = 0
        for i, group in enumerate(param_dicts):
            lr = group['lr']
            wd = group.get('weight_decay', args.weight_decay if hasattr(args, 'weight_decay') else 1e-4)
            param_num = sum(p.numel() for p in group['params'])
            total_trainable_params += param_num
            print(f"参数组{i + 1}: 学习率={lr}, 权重衰减={wd}, 参数数量={param_num}")
        print(f"总可训练参数数量: {total_trainable_params}")

    # ====================== 核心调整4：优化器初始化（兼容空参数组） ======================
    # 防止参数组为空
    if not param_dicts:
        raise ValueError("没有可训练的参数！请检查参数冻结逻辑或启用适配器")

    # 适配args：如果没有weight_decay参数，设置默认值
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 1e-4
    optim = torch.optim.AdamW(
        param_dicts,
        lr=args.lr_vit if hasattr(args, 'lr_vit') else 1e-5,
        weight_decay=weight_decay
    )

    # 学习率调度器（适配args）
    lr_drop = args.lr_drop if hasattr(args, 'lr_drop') else 20
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_drop)

    # ====================== 恢复训练状态 ======================
    if args.resume:
        try:
            optim.load_state_dict(checkpoint['optim_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch, iteration=iteration,
                                    scaler=scaler)
            if rank == 0:
                print(f"\n恢复训练：从epoch {epoch} 开始")
        except KeyError as e:
            if rank == 0:
                print(f"\n警告：检查点中缺少{e}，使用新的优化器/调度器")
                engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    # ====================== 保存参数配置 ======================
    import json
    args_dict = args.__dict__.copy()
    # 过滤不可序列化的参数
    for k in list(args_dict.keys()):
        if isinstance(args_dict[k], (torch.Tensor, np.ndarray)):
            del args_dict[k]
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args_dict, f, indent=2)
    f.close()

    # ====================== 启动训练 ======================
    if rank == 0:
        print(f"\n=== 启动训练 ===")
        print(f"总轮数: {args.epochs}")
        print(f"适配器启用状态: {args.use_text_adapter}")
        if args.use_text_adapter:
            print(f"适配器学习率: {lr_adapter}")
    engine(args.epochs)


if __name__ == '__main__':
    args = get_args()
    print(args)

    if args.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ["MASTER_PORT"] = args.port
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    print('WORLD_SIZE ' + str(os.environ.get("WORLD_SIZE", 1)))

    os.environ["MASTER_ADDR"] = "localhost"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank = local_rank

    # 新增：设置适配器默认参数（如果args中没有）
    if not hasattr(args, 'use_text_adapter'):
        args.use_text_adapter = True  # 默认启用适配器
    if not hasattr(args, 'lr_text_adapter'):
        args.lr_text_adapter = 5e-4  # 适配器默认学习率
    if not hasattr(args, 'text_adapter_dim'):
        args.text_adapter_dim = 64  # 适配器瓶颈层维度
    if not hasattr(args, 'lora_rank'):
        args.lora_rank = 2  # LoRA秩
    if not hasattr(args, 'lora_alpha'):
        args.lora_alpha = 16  # LoRA缩放系数

    if local_rank == 0:
        wandb.init(project='LAIN', name=args.output_dir)

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    main(local_rank, args)