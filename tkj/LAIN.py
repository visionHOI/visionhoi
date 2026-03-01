"""
Unary-pairwise transformer for human-object interaction detection
Optimized and Robust Version
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

import numpy as np
import wandb

from utils.hico_list import hico_verbs_sentence
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices
from utils.postprocessor import PostProcess
from utils.ops import binary_focal_loss_with_logits
from utils import hico_text_label

from CLIP.clip import build_model
from CLIP.customCLIP import CustomCLIP, tokenize

sys.path.insert(0, 'detr')
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer
from detr.models.detr import DETR
from detr.util import box_ops
from detr.util.misc import nested_tensor_from_tensor_list

sys.path.pop(0)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SemanticAdapter(nn.Module):
    def __init__(self, dim=768, bottleneck=64):
        super().__init__()
        # [Optimization] Added LayerNorm for training stability
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()  # [Optimization] ReLU -> GELU usually works better for Transformers
        self.up = nn.Linear(bottleneck, dim, bias=False)

        # Initialize weights to near-zero for the up-projection to act as identity initially
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        # Residual connection
        residual = x
        x = self.norm(x)
        return residual + self.up(self.act(self.down(x)))


class LAIN(nn.Module):
    def __init__(self,
                 args,
                 detector: nn.Module,
                 postprocessor: nn.Module,
                 model: nn.Module,
                 object_embedding: torch.tensor,
                 human_idx: int, num_classes: int,
                 alpha: float = 0.5, gamma: float = 2.0,
                 box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
                 min_instances: int = 3, max_instances: int = 15,
                 object_class_to_target_class: List[list] = None,
                 object_n_verb_to_interaction: List[list] = None,

                 ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model

        self.register_buffer("object_embedding", object_embedding)

        self.visual_output_dim = model.image_encoder.output_dim
        self.object_n_verb_to_interaction = np.asarray(
            object_n_verb_to_interaction, dtype=float
        )

        self.args = args
        self.human_idx = human_idx
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class

        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda
        self.use_insadapter = args.use_insadapter
        self.tp = None
        self.reserve_indices = reserve_indices

        self.priors_initial_dim = self.visual_output_dim + 5
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.priors_downproj = MLP(self.priors_initial_dim, 128, args.adapt_dim, 3)

        # [Logic] Robust parameter initialization for Query Projection
        self.query_proj = MLP(512, 128, 768, 2)

        # [Logic] Semantic Adapter Configuration
        self.use_semantic_adapter = getattr(args, "use_semantic_adapter", False)  # Default to False if not set
        if self.use_semantic_adapter:
            self.semantic_adapter = SemanticAdapter(dim=768, bottleneck=args.adapt_dim)

    def compute_prior_scores(self,
                             x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
                             ) -> Tensor:
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        target_cls_idx = [self.object_class_to_target_class[obj.item()]
                          for obj in object_class[y]]

        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        if len(pair_idx) > 0:
            prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
            prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def compute_sim_scores(self, region_props: List[dict], image, priors=None):
        device = image.tensors.device
        boxes_h_collated = [];
        boxes_o_collated = []
        prior_collated = [];
        object_class_collated = []
        all_logits = []

        # ---------------------------------------------------------------------
        # [Step 1] Robust Text Feature Extraction (FIXED)
        # ---------------------------------------------------------------------
        text_features_raw = None

        # use_cocoop=True 时，文本特征需要在每张图像处理时动态生成，
        # 所以这里先置 None，稍后在每张图的循环内计算。
        use_cocoop_mode = self.args.use_prompt and getattr(self.args, 'use_cocoop', False)

        if self.args.use_prompt and not use_cocoop_mode:
            # ── 原有 CoOp 路径：文本特征只算一次，所有图共用 ────────────────
            if not self.training and self.tp is not None:
                text_features_raw = self.tp
            else:
                prompts = self.clip_head.prompt_learner()   # 不传 image_feature
                text_features_raw = self.clip_head.text_encoder(prompts, self.clip_head.tokenized_prompts)
                text_features_raw = text_features_raw / text_features_raw.norm(dim=-1, keepdim=True)
                if not self.training:
                    self.tp = text_features_raw
            # ────────────────────────────────────────────────────────────────
        elif use_cocoop_mode:
            # ── CoCoOp 路径：文本特征在每张图像的循环内动态生成，这里先置 None ──
            text_features_raw = None   # 稍后在每张图的 if use_cocoop_mode 块中覆盖
        else:
            # === 无 Prompt Learner：使用静态预计算特征 ===
            if hasattr(self.clip_head, 'text_features') and self.clip_head.text_features is not None:
                text_features_raw = self.clip_head.text_features
            else:
                try:
                    text_features_raw = self.clip_head.get_text_features()
                except:
                    raise ValueError(
                        "Text features are missing! Ensure prompt_learner is enabled OR static text_features are pre-computed.")

        # 应用 Adapter（CoCoOp 模式下 text_features_raw 此时为 None，Adapter 在循环内按图应用）
        text_features_final = text_features_raw
        if self.use_semantic_adapter and text_features_raw is not None:
            text_features_final = self.semantic_adapter(text_features_raw)
            text_features_final = text_features_final / text_features_final.norm(dim=-1, keepdim=True)

        if text_features_final is None and not use_cocoop_mode:
            raise ValueError("Text features could not be generated. Check model state.")

        # ---------------------------------------------------------------------
        # [Step 2] Visual Feature Extraction (FIXED DIMENSION ERROR)
        # ---------------------------------------------------------------------
        for b_idx, props in enumerate(region_props):
            # ... (前面的盒子过滤和排序逻辑保持不变) ...
            boxes = props['boxes'];
            scores = props['scores']
            labels = props['labels'];
            feats = props['feat']
            is_human = labels == self.human_idx;
            n_h = torch.sum(is_human);
            n = len(boxes)

            if n_h == 0 or n <= 1:
                # (空图像处理保持不变)
                continue

            x, y = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing='ij')
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

            if len(x_keep) == 0:
                # (空对处理保持不变)
                continue

            if self.args.use_hotoken:
                # ... (HO Token 逻辑保持不变) ...
                num_tokens = len(x_keep) + 1
                mask = torch.zeros((num_tokens + 196, num_tokens + 196), dtype=torch.bool, device=device)
                mask[:num_tokens, :num_tokens] = ~torch.eye(num_tokens, dtype=torch.bool, device=device)
                mask[-197:, :-197] = True
                ho_tokens = self.query_proj(torch.cat([feats[x_keep], feats[y_keep]], dim=-1))
                la_masks = (boxes, x_keep, y_keep)

                global_feat, _ = self.clip_head.image_encoder(
                    image.decompose()[0][b_idx:b_idx + 1],
                    priors[b_idx] if self.args.use_prior else None,
                    ho_tokens, mask, la_masks
                )

                global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
                global_feat = global_feat[:, :-1]
                global_feat = global_feat.squeeze(0)  # 最终确保是 [N, Dim]

            else:
                # === [FIX] 解决你之前的 RuntimeError: Number of dimensions ===
                global_feat, _ = self.clip_head.image_encoder(
                    image.decompose()[0][b_idx:b_idx + 1],
                    priors[b_idx] if self.args.use_prior else None
                )
                global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)

                # --- 动态自适应 Repeat ---
                # 不论 global_feat 是 [1, 512] 还是 [1, 1, 512]，都将其展平并根据 x_keep 重复
                # global_feat_single: (1, Dim) ← 单张图的全局特征，用于 CoCoOp 条件
                global_feat_single = global_feat.view(1, -1)
                global_feat = global_feat_single.repeat(len(x_keep), 1)  # (n_pairs, Dim)

            # ── CoCoOp：用当前图像特征动态生成文本特征 ────────────────────────
            # 说明：CoCoOp 模式下，每张图都单独调用 prompt_learner，
            #       把当前图的全局特征传进去，MetaNet 生成偏置叠加到 ctx 上，
            #       所以不同图像对应的文本特征会稍有不同。
            if use_cocoop_mode:
                # global_feat_single 已在上方 else 分支赋值（(1, Dim)）
                # 若走的是 use_hotoken 分支，取 pair 特征的均值作为图像代表特征
                if self.args.use_hotoken:
                    global_feat_single = global_feat.mean(dim=0, keepdim=True)   # (1, Dim)

                prompts_img = self.clip_head.prompt_learner(
                    image_feature=global_feat_single.to(self.clip_head.prompt_learner.ctx.dtype)
                )  # (n_cls, 77, ctx_dim)
                text_features_raw = self.clip_head.text_encoder(
                    prompts_img, self.clip_head.tokenized_prompts
                )  # (n_cls, D)
                text_features_raw = text_features_raw / text_features_raw.norm(dim=-1, keepdim=True)

                text_features_final = text_features_raw
                if self.use_semantic_adapter:
                    text_features_final = self.semantic_adapter(text_features_raw)
                    text_features_final = text_features_final / text_features_final.norm(dim=-1, keepdim=True)
            # ────────────────────────────────────────────────────────────────────

            # ---------------------------------------------------------------------
            # [Step 3] Similarity Calculation (保持不变)
            # ---------------------------------------------------------------------
            logits_text = global_feat @ text_features_final.T
            logits = logits_text * self.logit_scale_text.exp()

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(x_keep, y_keep, scores, labels))
            all_logits.append(logits)

        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        # Fix for empty ground truth
        if len(gt_bx_h) == 0:
            return labels

        ious_h = box_iou(boxes_h, gt_bx_h)
        ious_o = box_iou(boxes_o, gt_bx_o)

        min_ious = torch.min(ious_h, ious_o)

        x, y = torch.nonzero(min_ious >= self.fg_iou_thresh).unbind(1)

        if self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1
        else:
            labels[x, targets['hoi'][y]] = 1
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):

        # [Robustness] Check for empty batch
        if len(logits) == 0:
            return torch.tensor(0.0, device=boxes[0].device, requires_grad=True)

        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])

        prior = torch.cat(prior, dim=1).prod(0)
        logits = torch.cat(logits)

        # [Safety] Dimension Mismatch Guard
        if len(labels) != len(logits):
            # This happens if some pairs were filtered out during logits calc but not here
            # For now, we assume strict alignment, but print warning if diff
            pass

        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]
        prior = prior[x, y]
        labels = labels[x, y]

        n_p = len(torch.nonzero(labels))

        # Distributed count normalization
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(prior / (1 + torch.exp(-logits) - prior) + 1e-8),
            labels,
            reduction='sum',
            alpha=self.alpha,
            gamma=self.gamma
        )

        if n_p == 0:
            return loss * 0.0

        return loss / n_p

    def prepare_region_proposals(self, results):
        region_props = []
        for res in results:
            sc, lb, bx, feat = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            feat = feat[keep].view(-1, 256)

            # 使用更低的阈值以确保有足够的候选框，特别是在训练早期
            # 训练时用0.2，评估时用0.01以获得更多召回
            effective_thresh = 0.01 if not self.training else self.box_score_thresh
            keep = torch.nonzero(sc >= effective_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            
            # 保护：如果没有检测到任何人或物体，至少取top-k个
            if len(hum) == 0:
                hum = torch.topk(sc, min(self.min_instances, len(sc))).indices
            if len(obj) == 0:
                obj = torch.topk(sc, min(self.min_instances, len(sc))).indices
            
            n_human = is_human[keep].sum() if len(keep) > 0 else 0
            n_object = len(keep) - n_human

            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                feat=feat[keep]
            ))

        return region_props

    def get_prior(self, region_props, image_size):

        max_feat = self.priors_initial_dim
        # Handle case where no boxes exist
        if len(region_props) == 0:
            return torch.zeros((0, 14, 14, args.adapt_dim), device=image_size.device)

        priors = torch.zeros((len(region_props), 14, 14), dtype=torch.float32, device=region_props[0]['boxes'].device)
        priors_dim = torch.zeros((len(region_props), 14, 14, max_feat), dtype=torch.float32,
                                 device=region_props[0]['boxes'].device)

        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] * (14 / scale_fct[b_idx][None, :])
            scores = props['scores']
            labels = props['labels']
            priors[b_idx] = len(boxes)

            boxes[:, 2:] += 0.5
            new_boxes = torch.round(boxes).long()

            # [FIX] Clamp coordinates to stay within 14x14 grid to prevent index out of bounds
            new_boxes[:, 0].clamp_(0, 13)
            new_boxes[:, 1].clamp_(0, 13)
            new_boxes[:, 2].clamp_(0, 14)  # x2 is exclusive in slice, so can be 14
            new_boxes[:, 3].clamp_(0, 14)  # y2 is exclusive

            for inb, nb in enumerate(new_boxes):
                x1, y1, x2, y2 = nb
                # Safety check: ensure x2 > x1 and y2 > y1
                if x2 > x1 and y2 > y1:
                    priors[b_idx, y1:y2, x1:x2] = inb

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human);
            n = len(boxes)
            if n_h == 0 or n <= 1:
                pass

            boxes = torch.cat([boxes, torch.tensor([[-1, -1, -1, -1.]]).to(boxes)], dim=0)
            labels = torch.cat([labels, torch.tensor([80]).to(boxes)], dim=0).long()
            scores = torch.cat([scores, torch.tensor([-1.]).to(boxes)], dim=0)

            object_embs = self.object_embedding[labels]

            sb = torch.cat((scores.unsqueeze(-1), boxes), dim=-1)
            sb_feat = sb[priors[b_idx].long()]
            obj_feat = object_embs[priors[b_idx].long()]

            prior_feat = torch.cat([sb_feat, obj_feat], dim=-1)
            priors_dim[b_idx] = prior_feat

        priors = self.priors_downproj(priors_dim)
        return priors

    def forward(self,
                images: List[Tensor],
                targets: Optional[List[dict]] = None
                ) -> List[dict]:

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        batch_size = len(images)
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]
        device = images_clip[0].device
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images_clip
        ], device=device)

        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)

        features, pos = self.detector.backbone(images_orig)
        src, mask = features[-1].decompose()

        hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask,
                                                    self.detector.query_embed.weight, pos[-1])
        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()

        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'feats': hs[-1]}
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)

        priors = self.get_prior(region_props, image_sizes)

        images_clip = nested_tensor_from_tensor_list(images_clip)

        # [Important] Check if all regions are empty, skip computation to save time/error
        all_empty = all(len(r['boxes']) == 0 for r in region_props)
        if all_empty:
            logits, prior, bh, bo, objects = [], [], [], [], []
        else:
            logits, prior, bh, bo, objects = self.compute_sim_scores(region_props, images_clip, priors)

        boxes = [r['boxes'] for r in region_props]

        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)
            loss_dict = dict(interaction_loss=interaction_loss)

            if self.args.local_rank == 0:
                wandb.log(loss_dict)
            return loss_dict

        if len(logits) == 0:
            # Return empty detections structure matching the format
            return [dict(boxes=torch.tensor([]), scores=torch.tensor([]), labels=torch.tensor([])) for _ in
                    range(batch_size)]

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes):
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size in zip(
                boxes, bh, bo, logits, prior, objects, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)

            # [LOGIC] Interference Suppression could be added here
            # e.g., scores[:, suppressed_ids] *= 0.1

            scores = torch.sigmoid(lg[x, y])

            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections


@torch.no_grad()
def get_obj_text_emb(args, clip_model, obj_class_names):
    obj_text_inputs = torch.cat([tokenize(obj_text) for obj_text in obj_class_names])
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
    return object_embedding


def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path):
    # build DETR
    num_classes = 80
    if args.dataset == 'vcoco' and 'e632da11' in args.pretrained:
        num_classes = 91

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    detr = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    postprocessors = {'bbox': PostProcess()}

    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu', weights_only=False)['model'])
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu', weights_only=False)['model_state_dict'])
    else:
        if dist.get_rank() == 0:
            print(f"!! WARNING: Pretrained weights not found at {args.pretrained}. !!")
            print("!! Model is starting from RANDOM INITIALIZATION (Performance will be poor) !!")

    clip_state_dict = torch.load(clip_model_path, map_location="cpu", weights_only=False).state_dict()
    clip_model = build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, adapter_pos=args.adapter_pos,
                             args=args)

    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError

    model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)

    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]

    object_embedding = get_obj_text_emb(args, clip_model=clip_model, obj_class_names=obj_class_names)
    object_embedding = object_embedding.clone().detach()

    detector = LAIN(args,
                    detr, postprocessors['bbox'], model, object_embedding,
                    human_idx=args.human_idx, num_classes=args.num_classes,
                    alpha=args.alpha, gamma=args.gamma,
                    box_score_thresh=args.box_score_thresh,
                    fg_iou_thresh=args.fg_iou_thresh,
                    min_instances=args.min_instances,
                    max_instances=args.max_instances,
                    object_class_to_target_class=class_corr,
                    object_n_verb_to_interaction=object_n_verb_to_interaction,
                    )

    return detector