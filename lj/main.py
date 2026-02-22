"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Complete Version for Kaggle/Colab with Semantic Adapter & Speed Control
"""
import os
import torch
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import sys
import argparse

from models.LAIN import build_detector
from utils.args import get_args

from engine import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

warnings.filterwarnings("ignore")


# --- 核心修复：限流加载器，代理原始 loader 的所有属性 ---
class LimitedLoader:
    def __init__(self, loader, limit):
        self.loader = loader
        self.limit = int(limit)
        self.dataset = loader.dataset

    def __iter__(self):
        count = 0
        for item in self.loader:
            if count >= self.limit:
                break
            yield item
            count += 1

    def __len__(self):
        return min(len(self.loader), self.limit)

    def __getattr__(self, name):
        # 这一步极其重要：当访问 sampler 时，转发给原始 loader
        return getattr(self.loader, name)


class FilterStream:
    def __init__(self, target):
        self.target = target

    def write(self, s):
        if "tensor(0" in s: return
        self.target.write(s)

    def flush(self):
        self.target.flush()


sys.stdout = FilterStream(sys.stdout)


def main(rank, args):
    # 分布式初始化
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(rank)

    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'

    if rank == 0:
        print("-" * 50)
        print(f"🚀 [CONFIG] Use Semantic Adapter: {getattr(args, 'use_semantic_adapter', False)}")
        print(f"🚀 [CONFIG] Adapter Dim: {getattr(args, 'adapt_dim', 64)}")
        if getattr(args, 'limit_batches', 0) > 0:
            print(f"⚡ [SPEED MODE] Training limited to {args.limit_batches} batches per epoch!")
        print("-" * 50)

    # 数据集加载
    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
                           clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
                           num_classes=args.num_classes, args=args)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
                          clip_model_name=args.clip_model_name, args=args)

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
    )

    # 应用限流
    if getattr(args, 'limit_batches', 0) > 0:
        train_loader = LimitedLoader(train_loader, args.limit_batches)

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=False)
    )

    # 构建模型
    args.human_idx = 0
    lain = build_detector(args, trainset.dataset.object_class_to_target_class,
                          object_n_verb_to_interaction=trainset.dataset.object_n_verb_to_interaction,
                          clip_model_path=args.clip_dir_vit)

    if args.dataset == 'hicodet' and args.eval:
        lain.object_class_to_target_class = testset.dataset.object_class_to_target_class

    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: Loading checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        lain.load_state_dict(checkpoint['model_state_dict'], strict=False)

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

    if args.eval:
        lain.eval()
        ap = engine.test_hico(test_loader, args)
        # 完整保留评估打印逻辑
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        if len(ap) == args.num_classes:
            print(
                f"mAP: {ap.mean() * 100:.2f}, rare: {ap[rare].mean() * 100:.2f}, non-rare: {ap[non_rare].mean() * 100:.2f}")
        if args.zs:
            zs_hoi_idx = hico_unseen_index[args.zs_type]
            ap_unseen = torch.as_tensor([ap[i] for i in zs_hoi_idx]).mean()
            print(f">>> Zero-shot ({args.zs_type}) | Unseen mAP: {ap_unseen * 100:.2f}")
        return

    # --- 训练参数配置 ---
    # 冻结基础检测器
    for p in lain.detector.parameters(): p.requires_grad = False

    # 配置 CLIP Head & Adapter
    for n, p in lain.named_parameters():
        if 'semantic_adapter' in n:
            p.requires_grad = True if getattr(args, 'use_semantic_adapter', False) else False
        elif 'clip_head' in n:
            # 开启 prompt 和 adaptermlp 权重
            if any(x in n for x in ['adaptermlp', 'prompt_learner', 'visual_prompt', 'visual.proj']):
                p.requires_grad = True
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True

    # 打印可训练参数 (Kaggle 环境下很有用，防止跑错)
    if rank == 0:
        print("\n[TRAINABLE PARAMETERS]")
        for n, p in lain.named_parameters():
            if p.requires_grad: print(f"  {n}")


    # 优化器
    param_dicts = [{"params": [p for n, p in lain.named_parameters() if p.requires_grad]}]
    optim = torch.optim.AdamW(param_dicts, lr=args.lr_vit, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)

    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    # 运行训练
    engine(args.epochs)


if __name__ == '__main__':
    args = get_args()

    # --- 环境变量补丁 (Kaggle/Colab 通用) ---
    if not hasattr(args, 'use_semantic_adapter'):
        setattr(args, 'use_semantic_adapter', False)
    if '--use_semantic_adapter' in sys.argv:
        args.use_semantic_adapter = True

    # 限流设置
    limit_env = os.environ.get("LIMIT_BATCHES", "0")
    setattr(args, 'limit_batches', int(limit_env))

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["MASTER_ADDR"] = "localhost"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank = local_rank

    if local_rank == 0:
        wandb.init(project='LAIN', mode="disabled")

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    main(local_rank, args)
