import os
import subprocess
import sys
import glob

# ================= Kaggle 配置区域 =================
KAGGLE_INPUT = "/kaggle/input/hico-20160224-det/hico_20160224_det/hico_20160224_det"
KAGGLE_WEIGHTS = "/kaggle/input/lain-pretrained-weights"

DATA_ROOT = "/kaggle/working/hicodet"
PRETRAINED = os.path.join(KAGGLE_WEIGHTS, "detr-r50-hicodet.pth")

# --- 关键修改：降低 BS 以防止 OOM，缩短 Epochs 确保 12 小时内能出 mAP ---
EPOCHS = 3  # 建议先跑 3 个 Epoch 验证 Adapter 有效性
BATCH_SIZE = 2  # 16G 显存下，使用 ViT-B 跑分布式/多任务建议设为 2
LR_HEAD = "1e-4"
# =================================================

experiments = [
    {
        "name": "Baseline_No_Adapter",
        "args": "--use_prompt",
        "output_dir": "/kaggle/working/checkpoints/baseline"
    },
    {
        "name": "With_Semantic_Adapter",
        "args": "--use_prompt --use_semantic_adapter --adapt_dim 64",
        "output_dir": "/kaggle/working/checkpoints/adapter_exp"
    }
]


def setup_kaggle_environment():
    """路径对齐与软链接"""
    print("\n🚀 正在初始化 Kaggle 环境...")
    inner_path = os.path.join(DATA_ROOT, "hico_20160224_det")
    os.makedirs(inner_path, exist_ok=True)

    drive_img_path = os.path.join(KAGGLE_INPUT, "images")
    local_img_path = os.path.join(inner_path, "images")

    if not os.path.exists(local_img_path):
        print(f"🔗 链接图片库...")
        subprocess.run(f"ln -sf {drive_img_path} {local_img_path}", shell=True)

    print("📝 拷贝标注文件...")
    subprocess.run(f"cp {KAGGLE_INPUT}/*.json {inner_path}/", shell=True)
    subprocess.run(f"cp {KAGGLE_INPUT}/*.json {DATA_ROOT}/", shell=True)


def run_cmd(cmd):
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "12345"
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    # 显存碎片优化策略
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"\n🔥 正在执行: {cmd}\n")
    process = subprocess.Popen(cmd, shell=True, env=env, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()


def main():
    setup_kaggle_environment()

    for exp in experiments:
        print(f"\n🌟 === 正在开始实验: {exp['name']} === 🌟")
        os.makedirs(exp['output_dir'], exist_ok=True)

        # 1. 训练命令
        # 修改点：backbone 设为 clip_ViT-B-16，手动添加 hidden_dim 512
        train_cmd = f"python main.py --dataset hicodet --batch-size {BATCH_SIZE} --epochs {EPOCHS} " \
                    f"--num-workers 4 --lr-head {LR_HEAD} --backbone resnet50 " \
                    f"--data-root {DATA_ROOT} --pretrained {PRETRAINED} " \
                    f"--output-dir {exp['output_dir']} --zs --zs_type rare_first {exp['args']}"

        run_cmd(train_cmd)

        # 2. 自动评估命令 (mAP)
        # 寻找训练产生的最后一个 checkpoint
        ckpts = glob.glob(os.path.join(exp['output_dir'], "*.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getctime)
            print(f"📊 训练完成，正在评估 {exp['name']} 的 mAP... (使用权重: {latest_ckpt})")

            eval_cmd = f"python main.py --dataset hicodet --data-root {DATA_ROOT} " \
                       f"--backbone resnet50 " \
                       f"--resume {latest_ckpt} --eval --output-dir {exp['output_dir']}"
            run_cmd(eval_cmd)

        # 3. 每个实验结束后强制释放显存
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()