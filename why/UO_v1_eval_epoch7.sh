# LAIN V1 Evaluation Script - Epoch 7
# UO (Unseen Object) Setting

CUDA_VISIBLE_DEVICES=0 python main_v1.py \
                              --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
                              --dataset hicodet --num-workers 6 --num_classes 117 --zs --zs_type unseen_object --output-dir checkpoints/UO_v1 \
                              --use_hotoken --use_prompt --N_CTX 36 --CSC --use_exp \
                              --use_insadapter --adapt_dim 32 --use_prior \
                              --use_scene_gate --scene_gate_type image \
                              --resume checkpoints/UO_v1/ckpt_27265_07.pt --eval --debug --port 11547
