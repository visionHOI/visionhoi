import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='LAIN: Language-Aware Instance Network for HOI Detection')

    # ====================== 基础优化器/Transformer参数 ======================
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr-drop', default=10, type=int,
                        help='Epoch interval for learning rate drop')
    parser.add_argument('--clip-max-norm', default=0.1, type=float,
                        help='Max norm for gradient clipping')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='Backbone network (resnet50/resnet101)')
    parser.add_argument('--dilation', action='store_true',
                        help='Use dilation in backbone')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help='Type of positional embedding')

    parser.add_argument('--repr-dim', default=512, type=int,
                        help='Representation dimension for HOI features')
    parser.add_argument('--hidden-dim', default=256, type=int,
                        help='Hidden dimension of Transformer')
    parser.add_argument('--enc-layers', default=6, type=int,
                        help='Number of encoder layers in Transformer')
    parser.add_argument('--dec-layers', default=6, type=int,
                        help='Number of decoder layers in Transformer')
    parser.add_argument('--dim-feedforward', default=2048, type=int,
                        help='Dimension of feedforward network in Transformer')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate in Transformer')
    parser.add_argument('--nheads', default=8, type=int,
                        help='Number of attention heads in Transformer')
    parser.add_argument('--num-queries', default=100, type=int,
                        help='Number of query slots in DETR')
    parser.add_argument('--pre-norm', action='store_true',
                        help='Use pre-normalization in Transformer')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false',
                        help='Disable auxiliary decoding losses (loss at each layer)')
    parser.add_argument('--set-cost-class', default=1, type=float,
                        help='Classification loss coefficient in set prediction')
    parser.add_argument('--set-cost-bbox', default=5, type=float,
                        help='Bounding box loss coefficient in set prediction')
    parser.add_argument('--set-cost-giou', default=2, type=float,
                        help='GIoU loss coefficient in set prediction')
    parser.add_argument('--bbox-loss-coef', default=5, type=float,
                        help='Bounding box loss coefficient')
    parser.add_argument('--giou-loss-coef', default=2, type=float,
                        help='GIoU loss coefficient')
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # ====================== 训练控制参数 ======================
    parser.add_argument('--cache', action='store_true',
                        help='Cache dataset features for fast evaluation')
    parser.add_argument('--box-score-thresh', default=0.2, type=float,
                        help='Bounding box score threshold for filtering')
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float,
                        help='IoU threshold for foreground objects')
    parser.add_argument('--min-instances', default=3, type=int,
                        help='Minimum number of instances per batch')
    parser.add_argument('--max-instances', default=15, type=int,
                        help='Maximum number of instances per batch')

    # ====================== CLIP ViT Model Parameters ======================
    # ViT-L/14@336px (uncomment to use)
    # parser.add_argument('--clip_visual_layers_vit', default=24, type=int)
    # parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    # parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    # parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    # parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)

    # ViT-B-16 (default)
    parser.add_argument('--clip_visual_layers_vit', default=12, type=int,
                        help='Number of visual layers in CLIP ViT')
    parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int,
                        help='Output dimension of CLIP visual encoder')
    parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int,
                        help='Input resolution for CLIP visual encoder')
    parser.add_argument('--clip_visual_width_vit', default=768, type=int,
                        help='Width of CLIP visual transformer')
    parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int,
                        help='Patch size of CLIP visual encoder')

    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int,
                        help='Width of CLIP text transformer')
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int,
                        help='Number of attention heads in CLIP text transformer')
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int,
                        help='Number of layers in CLIP text transformer')
    parser.add_argument('--clip_text_context_length_vit', default=77, type=int,
                        help='Context length of CLIP text encoder')

    # ====================== 特征掩码/采样参数 ======================
    parser.add_argument('--feat_mask_type', type=int, default=0,
                        help='0: dropout(random mask); 1: None (no mask)')
    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Apply repeat factor sampling for tail categories')
    parser.add_argument('--dataset_file', default='coco', type=str,
                        help='Dataset file format (coco for HOI)')

    # ====================== Deformable DETR Parameters ======================
    parser.add_argument('--d_detr', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Whether to use Deformable DETR')
    parser.add_argument('--lr_backbone', default=2e-5, type=float,
                        help='Learning rate for backbone network')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--with_box_refine', default=False, action='store_true',
                        help='Use box refinement in Deformable DETR')
    parser.add_argument('--two_stage', default=False, action='store_true',
                        help='Use two-stage Deformable DETR')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="Position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int,
                        help='Number of feature levels in Deformable DETR')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='Number of sampling points in decoder')
    parser.add_argument('--enc_n_points', default=4, type=int,
                        help='Number of sampling points in encoder')
    parser.add_argument('--mask_loss_coef', default=1, type=float,
                        help='Mask loss coefficient')
    parser.add_argument('--dice_loss_coef', default=1, type=float,
                        help='Dice loss coefficient')
    parser.add_argument('--cls_loss_coef', default=2, type=float,
                        help='Classification loss coefficient')
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='Alpha for focal loss')

    # ====================== Learning Rate Parameters ======================
    parser.add_argument('--lr-head', default=1e-3, type=float,
                        help='Learning rate for head layers')
    parser.add_argument('--lr-vit', default=1e-3, type=float,
                        help='Learning rate for CLIP ViT layers')

    # ====================== Zero-Shot HOI Parameters ======================
    parser.add_argument('--zs', action='store_true',
                        help='Whether to use zero-shot setting for HOI detection')
    parser.add_argument('--zs_type', type=str, default='rare_first',
                        choices=['rare_first', 'non_rare_first', 'unseen_verb', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4',
                                 'unseen_object'],
                        help='Zero-shot split type for HICO-DET')

    # ====================== Dataset Parameters ======================
    parser.add_argument('--dataset', default='hicodet', type=str,
                        choices=['hicodet', 'vcoco'],
                        help='Dataset name (hicodet/vcoco)')
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str,
                        help='Train/test partitions (train2015/test2015 for HICO-DET)')
    parser.add_argument('--num_classes', type=int, default=117,
                        help='Number of HOI classes (117 for HICO-DET, 24 for V-COCO)')
    parser.add_argument('--data-root', default='./hicodet', type=str,
                        help='Root path of the dataset')

    # ====================== Training/Evaluation Parameters ======================
    parser.add_argument('--epochs', default=30, type=int,  # 适配Complex模式，默认30轮
                        help='Number of training epochs')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--num-workers', default=2, type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--eval', action='store_true',
                        help='Only evaluate the model (no training)')

    # ====================== Model Path Parameters ======================
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str,
                        help='Path to CLIP ViT model weights')
    parser.add_argument('--pretrained', default='', type=str,
                        help='Path to pretrained DETR detector weights')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume training from checkpoint')
    parser.add_argument('--output-dir', default='checkpoints', type=str,
                        help='Directory to save checkpoints and logs')

    # ====================== HOI Core Parameters ======================
    parser.add_argument('--use_hotoken', action='store_true',
                        help='Use human-object (HO) tokens for HOI representation')
    parser.add_argument('--use_prior', action='store_true',
                        help='Use prior scores for HOI detection')
    parser.add_argument('--use_exp', action='store_true',
                        help='Use exponential scaling for prior scores')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Alpha for focal loss')
    parser.add_argument('--gamma', default=0.2, type=float,
                        help='Gamma for focal loss')
    parser.add_argument('--hyper_lambda', type=float, default=2.8,
                        help='Scaling factor for prior scores during inference')

    # ====================== Instance Adapter Parameters (Visual Side) ======================
    parser.add_argument('--use_insadapter', action='store_true',
                        help='Use instance adapter in CLIP visual encoder')
    parser.add_argument('--adapter_num_layers', type=int, default=1,
                        help='Number of instance adapter layers')
    parser.add_argument('--adapt_dim', default=32, type=int,
                        help='Dimension of instance adapter hidden layer')
    parser.add_argument('--adapter_alpha', default=1., type=float,
                        help='Scaling factor for instance adapter output')
    parser.add_argument('--adapter_pos', type=str, default='all',
                        choices=['all', 'front', 'end', 'random', 'last', '03', '47', '811'],
                        help='Position of instance adapter layers in CLIP')
    parser.add_argument('--adapter_scalar', default='learnable_scalar', type=str,
                        help='Type of scalar for instance adapter (learnable/fixed)')

    # ====================== Prompt Learning Parameters ======================
    parser.add_argument('--use_prompt', action='store_true',
                        help='Use prompt learning in CLIP text encoder')
    parser.add_argument('--N_CTX', type=int, default=24,
                        help='Number of context vectors for prompt learning')
    parser.add_argument('--CSC', action='store_true',
                        help='Use class-specific context (CSC) for prompt learning')
    parser.add_argument('--CTX_INIT', type=str, default='',
                        help='Initialization words for context vectors')
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end',
                        choices=['middle', 'end', 'front'],
                        help='Position of class token in prompt')

    # ====================== Miscellaneous Parameters ======================
    parser.add_argument('--job_id', default=1985, type=int,
                        help='Job ID for logging')
    parser.add_argument('--vis', action='store_true',
                        help='Visualize detection results')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug mode (more verbose logs)')
    parser.add_argument('--seed', default=66, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device for training/testing (cuda/cpu)')
    parser.add_argument('--port', default='7894', type=str,
                        help='Port for distributed training')
    parser.add_argument('--print-interval', default=100, type=int,
                        help='Interval to print training logs')
    parser.add_argument('--world-size', default=1, type=int,
                        help='Number of processes for distributed training')

    # ====================== Text Adapter Parameters (Text Side) ======================
    parser.add_argument('--use_text_adapter', action='store_true',
                        help='Enable text semantic adapter in CLIP text encoder')
    parser.add_argument('--text_adapter_dim', type=int, default=64,
                        help='Bottleneck dimension of text adapter')
    parser.add_argument('--adapter_mode', type=str, default='simple', choices=['simple', 'complex'],
                        help='Text adapter mode (simple/complex)')
    parser.add_argument('--lr-adapter', default=1e-3, type=float,
                        help='Learning rate for simple text adapter (recommend 1e-3)')
    parser.add_argument('--lr-adapter-complex', default=5e-4, type=float,
                        help='Learning rate for complex text adapter (LoRA layers, recommend 5e-4)')
    parser.add_argument('--lr-gate', default=1e-4, type=float,
                        help='Learning rate for gate layer in complex text adapter (recommend 1e-4)')
    parser.add_argument('--lora-rank', default=8, type=int,  # 修正：删除重复的--lora_rank
                        help='Rank for LoRA in complex text adapter')
    parser.add_argument('--lora-alpha', default=16, type=float,
                        help='Alpha scaling factor for LoRA in complex text adapter')

    return parser.parse_args()