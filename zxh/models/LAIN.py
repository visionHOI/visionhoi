"""
Unary-pairwise transformer for human-object interaction detection
Integrated version: Robust + SceneGate + TextSemanticAdapter
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

# ========== SceneGate ==========
from models.scene_gate import build_scene_gate

# ========== Text Semantic Adapter ==========
from models.lora_utils import TextSemanticAdapter

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
    """ Simple residual adapter (from LAIN.py) """
    def __init__(self, dim=768, bottleneck=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
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
        self.object_n_verb_to_interaction = np.asarray(object_n_verb_to_interaction, dtype=float)

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

        # HO token projection
        self.query_proj = MLP(512, 128, 768, 2)

        # ========== SceneGate ==========
        self.use_scene_gate = getattr(args, 'use_scene_gate', False)
        self.scene_gate_type = getattr(args, 'scene_gate_type', 'image')
        if self.use_scene_gate:
            self.scene_gate = build_scene_gate(
                gate_type=self.scene_gate_type,
                dim=self.visual_output_dim,
                hidden_dim=128
            )
            if dist.get_rank() == 0:
                print(f"[INFO] SceneGate initialized with type: {self.scene_gate_type}")

        # ========== Text Semantic Adapter (LAINv2) ==========
        self.use_text_adapter = getattr(args, 'use_text_adapter', False)
        if self.use_text_adapter:
            clip_text_dim = getattr(self.clip_head.text_encoder, 'output_dim', 512)
            self.text_adapter = TextSemanticAdapter(
                dim=clip_text_dim,
                bottleneck=getattr(args, 'text_adapter_dim', 64),
                rank=getattr(args, 'lora_rank', 2),
                alpha=getattr(args, 'adapter_alpha', 8),
                adapter_mode=getattr(args, 'adapter_mode', 'simple')
            )
            self.use_ho_align = getattr(args, 'use_ho_align', True)
            if dist.get_rank() == 0:
                print(f"[INFO] TextSemanticAdapter | mode={args.adapter_mode} rank={args.lora_rank}")

        # ========== Simple Semantic Adapter (fallback, from LAIN.py) ==========
        self.use_semantic_adapter = getattr(args, 'use_semantic_adapter', False)
        if self.use_semantic_adapter and not self.use_text_adapter:
            self.semantic_adapter = SemanticAdapter(dim=768, bottleneck=args.adapt_dim)
            if dist.get_rank() == 0:
                print("[INFO] Simple SemanticAdapter enabled")

        # ========== Dimension projector (if visual/text dims mismatch) ==========
        self.dim_proj = None

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
        boxes_h_collated = []
        boxes_o_collated = []
        prior_collated = []
        object_class_collated = []
        all_logits = []

        # ---------------------------------------------------------------------
        # Text feature extraction (robust version from LAIN.py)
        # ---------------------------------------------------------------------
        text_features_raw = None
        use_cocoop_mode = self.args.use_prompt and getattr(self.args, 'use_cocoop', False)

        if self.args.use_prompt and not use_cocoop_mode:
            if not self.training and self.tp is not None:
                text_features_raw = self.tp
            else:
                prompts = self.clip_head.prompt_learner()
                text_features_raw = self.clip_head.text_encoder(prompts, self.clip_head.tokenized_prompts)
                text_features_raw = text_features_raw / text_features_raw.norm(dim=-1, keepdim=True)
                if not self.training:
                    self.tp = text_features_raw
        elif use_cocoop_mode:
            text_features_raw = None   # will be computed per image
        else:
            # Static text features (pre-computed)
            if hasattr(self.clip_head, 'text_features') and self.clip_head.text_features is not None:
                text_features_raw = self.clip_head.text_features
            else:
                try:
                    text_features_raw = self.clip_head.get_text_features()
                except:
                    raise ValueError("Text features are missing!")

        # ---------------------------------------------------------------------
        # Per-image processing
        # ---------------------------------------------------------------------
        for b_idx, props in enumerate(region_props):
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            feats = props['feat']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human)
            n = len(boxes)

            if n_h == 0 or n <= 1:
                # Empty image: push empty placeholders
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Reorder: humans first
            if not torch.all(labels[:n_h] == self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]
                scores = scores[perm]
                labels = labels[perm]
                feats = feats[perm]

            x, y = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing='ij')
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                raise ValueError("There are no valid human-object pairs")

            # -----------------------------------------------------------------
            # Visual feature extraction
            # -----------------------------------------------------------------
            ho_feat_global = None
            if self.args.use_hotoken:
                # Build HO tokens
                num_tokens = len(x_keep) + 1   # +1 for CLS
                mask = torch.zeros((num_tokens + 196, num_tokens + 196), dtype=torch.bool, device=device)
                mask[:num_tokens, :num_tokens] = ~torch.eye(num_tokens, dtype=torch.bool, device=device)
                mask[-197:, :-197] = True

                ho_tokens = self.query_proj(torch.cat([feats[x_keep], feats[y_keep]], dim=-1))
                la_masks = (boxes, x_keep, y_keep)

                # Call image encoder with HO tokens (returns three values)
                ho_tokens_out, cls_token, local_feat = self.clip_head.image_encoder(
                    image.decompose()[0][b_idx:b_idx + 1],
                    priors[b_idx] if self.args.use_prior else None,
                    ho_tokens,
                    mask, la_masks
                )

                # Normalize
                ho_tokens_out = ho_tokens_out / ho_tokens_out.norm(dim=-1, keepdim=True)
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

                # SceneGate fusion
                if self.use_scene_gate:
                    fused_tokens = self.scene_gate(ho_tokens_out, cls_token)
                else:
                    fused_tokens = ho_tokens_out

                # Global HO feature for text adapter alignment
                ho_feat_global = ho_tokens.mean(dim=0, keepdim=True)   # (1, dim)

                # Use fused tokens as visual representation for similarity
                visual_feat = fused_tokens   # (1, num_tokens, dim)
                # Remove CLS token if present (first token)
                if visual_feat.shape[1] > len(x_keep):
                    visual_feat = visual_feat[:, :-1, :]   # keep only HO tokens
            else:
                # Standard forward (no HO tokens)
                global_feat, local_feat = self.clip_head.image_encoder(
                    image.decompose()[0][b_idx:b_idx + 1],
                    priors[b_idx] if self.args.use_prior else None
                )
                global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
                # global_feat shape: (1, dim) or (1, 1, dim)
                global_feat = global_feat.view(1, -1)               # (1, dim)
                visual_feat = global_feat.repeat(len(x_keep), 1)    # (n_pairs, dim)

            # -----------------------------------------------------------------
            # Text feature adaptation (if needed)
            # -----------------------------------------------------------------
            text_features = text_features_raw
            if use_cocoop_mode:
                # CoCoOp: generate text features conditioned on current image
                if self.args.use_hotoken:
                    # Use mean of HO tokens as image condition
                    img_cond = ho_feat_global.to(self.clip_head.prompt_learner.ctx.dtype)
                else:
                    img_cond = visual_feat.mean(dim=0, keepdim=True).to(self.clip_head.prompt_learner.ctx.dtype)
                prompts_img = self.clip_head.prompt_learner(image_feature=img_cond)
                text_features = self.clip_head.text_encoder(prompts_img, self.clip_head.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Apply text adapters
            if self.use_text_adapter and text_features is not None:
                # Pass global HO feature for alignment if available
                align_feat = ho_feat_global if self.use_ho_align else None
                text_features = self.text_adapter(text_features, align_feat)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            elif self.use_semantic_adapter and text_features is not None:
                text_features = self.semantic_adapter(text_features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if text_features is None:
                raise RuntimeError("No text features available")

            # -----------------------------------------------------------------
            # Dimension alignment (if needed)
            # -----------------------------------------------------------------
            if visual_feat.shape[-1] != text_features.shape[-1]:
                if self.dim_proj is None or self.dim_proj.weight.shape[0] != text_features.shape[-1]:
                    self.dim_proj = nn.Linear(text_features.shape[-1], visual_feat.shape[-1], device=device)
                    nn.init.eye_(self.dim_proj.weight)
                text_features = self.dim_proj(text_features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # -----------------------------------------------------------------
            # Similarity computation
            # -----------------------------------------------------------------
            logits_text = visual_feat @ text_features.T
            logits = logits_text * self.logit_scale_text.exp()

            # Collect outputs
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

        if len(gt_bx_h) == 0:
            return labels

        ious_h = box_iou(boxes_h, gt_bx_h)
        ious_o = box_iou(boxes_o, gt_bx_o)
        min_ious = torch.min(ious_h, ious_o)

        x, y = torch.nonzero(min_ious >= self.fg_iou_thresh).unbind(1)

        if self.num_classes in [117, 24, 407]:
            labels[x, targets['labels'][y]] = 1
        else:
            labels[x, targets['hoi'][y]] = 1
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
        if len(logits) == 0:
            return torch.tensor(0.0, device=boxes[0].device, requires_grad=True)

        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])

        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)

        if len(x) == 0:
            return torch.tensor(0.0, device=prior.device, requires_grad=True)

        logits = torch.cat(logits)
        logits = logits[x, y]
        prior = prior[x, y]
        labels = labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        if n_p == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = binary_focal_loss_with_logits(
            torch.log(prior / (1 + torch.exp(-logits) - prior) + 1e-8),
            labels,
            reduction='sum',
            alpha=self.alpha,
            gamma=self.gamma
        )
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

            effective_thresh = 0.01 if not self.training else self.box_score_thresh
            keep = torch.nonzero(sc >= effective_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)

            # Protect against empty detections
            if len(hum) == 0:
                hum = torch.topk(sc, min(self.min_instances, len(sc))).indices
            if len(obj) == 0:
                obj = torch.topk(sc, min(self.min_instances, len(sc))).indices

            n_human = is_human[keep].sum() if len(keep) > 0 else 0
            n_object = len(keep) - n_human

            # Human selection
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            # Object selection
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
        if len(region_props) == 0:
            return torch.zeros((0, 14, 14, self.args.adapt_dim), device=image_size.device)

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
            new_boxes[:, 0].clamp_(0, 13)
            new_boxes[:, 1].clamp_(0, 13)
            new_boxes[:, 2].clamp_(0, 14)
            new_boxes[:, 3].clamp_(0, 14)

            for inb, nb in enumerate(new_boxes):
                x1, y1, x2, y2 = nb
                if x2 > x1 and y2 > y1:
                    priors[b_idx, y1:y2, x1:x2] = inb

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human)
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
        image_sizes = torch.as_tensor([im.size()[-2:] for im in images_clip], device=device)

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
            return [dict(boxes=torch.tensor([]), scores=torch.tensor([]), labels=torch.tensor([])) for _ in range(batch_size)]

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
