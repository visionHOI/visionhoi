id=12345
port=29500

CUDA_VISIBLE_DEVICES=0 \
torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=1 --rdzv_endpoint=127.0.0.1:$port \
mainv2.py \
--pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth \
--clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
--output-dir checkpoints/NF1_complex_optim \
--dataset hicodet --zs --zs_type non_rare_first --num_classes 117 \
--epochs 30 --batch-size 4 --num-workers 4 \
--use_text_adapter --adapter_mode complex --text_adapter_dim 64 \
--lora-rank 4 --lora-alpha 16 --lr-adapter-complex 5e-4 \
--lr-vit 1e-5 --lr-head 1e-4 --weight-decay 1e-4 --lr-drop 10 \
--use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
--use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1.0 \
--print-interval 100