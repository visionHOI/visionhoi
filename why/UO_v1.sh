# LAIN V1 Training Script with SceneGate
# UO (Unseen Object) Setting

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300
CUDA_VISIBLE_DEVICES=0

torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main_v1.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --output-dir checkpoints/UO_v1 \
         --dataset hicodet --zs --zs_type unseen_object --num_classes 117 --num-workers 4 \
         --epochs 20 --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
         --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1. \
         --use_scene_gate --scene_gate_type image --scene_gate_hidden_dim 128 \
         --print-interval 100
