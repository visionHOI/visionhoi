# LAIN V1 Training Script with Both SceneGate and Semantic Adapter
# UO (Unseen Object) Setting

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=${GPU_NUM:-1}
num_workers=${NUM_WORKERS:-4}
batch_size=${BATCH_SIZE:-8}
WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}

# Do not hard-limit GPU visibility here.
# If needed, set CUDA_VISIBLE_DEVICES from shell before launching.

torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --output-dir checkpoints/UO_v1_integrated \
         --dataset hicodet --data-root /root/LAIN-origin/LAIN-main/hicodet --zs --zs_type unseen_object --num_classes 117 --num-workers $num_workers \
         --epochs 70 --batch-size $batch_size --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
         --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1. \
         --use_scene_gate --scene_gate_type image --scene_gate_hidden_dim 128 \
         --use_text_adapter --text_adapter_dim 64 --lora_rank 4 --lora_alpha 16 --lr_text_adapter 5e-4 \
         --print-interval 100
