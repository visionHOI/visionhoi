"""
Unary-pairwise transformer for human-object interaction detection
V1: Added SceneGate for CLS token fusion

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
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

from CLIP.clip_v1 import build_model
from CLIP.customCLIP import CustomCLIP, tokenize
from models.scene_gate import build_scene_gate

sys.path.insert(0, 'detr')
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer
from detr.models.detr import DETR
from detr.util import box_ops
from detr.util.misc import nested_tensor_from_tensor_list
sys.path.pop(0)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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

        self.register_buffer("object_embedding",object_embedding)

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

        self.num_classes = num_classes

        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda

        self.use_insadapter = args.use_insadapter
        self.tp = None
        self.reserve_indices = reserve_indices


        self.priors_initial_dim = self.visual_output_dim + 5
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.priors_downproj = MLP(self.priors_initial_dim, 128, args.adapt_dim, 3)


        self.query_proj = MLP(512, 128, 768, 2)
        
        # ========== V1 MODIFICATION: 添加 SceneGate 模块 ==========
        self.use_scene_gate = getattr(args, 'use_scene_gate', True)
        self.scene_gate_type = getattr(args, 'scene_gate_type', 'image')
        
        if self.use_scene_gate:
            self.scene_gate = build_scene_gate(
                gate_type=self.scene_gate_type,
                dim=self.visual_output_dim,
                hidden_dim=128
            )
            print(f"[INFO] SceneGate initialized with type: {self.scene_gate_type}")
        # =========================================================

    def _reset_parameters(self):
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def compute_sim_scores(self, region_props: List[dict], image, priors=None):
        device = image.tensors.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        all_logits = []

        text_features = None
        if self.args.use_prompt:
            if not self.training:
                if self.tp is None:
                    prompts = self.clip_head.prompt_learner()
                    text_features = self.clip_head.text_encoder(prompts, self.clip_head.tokenized_prompts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    self.tp = text_features
                else:
                    text_features = self.tp
            else:
                prompts = self.clip_head.prompt_learner()
                text_features = self.clip_head.text_encoder(prompts, self.clip_head.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            feats = props['feat']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                raise ValueError("There are no valid human-object pairs")

            if self.args.use_hotoken:
                num_tokens = len(x_keep) + 1

                mask = torch.zeros((num_tokens + 196, num_tokens + 196), dtype=torch.bool, device=device)
                mask[:num_tokens, :num_tokens] = ~torch.eye(num_tokens, dtype=torch.bool, device=device)
                mask[-197:, :-197] = True

                ho_tokens = self.query_proj(torch.cat([feats[x_keep],feats[y_keep]],dim=-1))

                la_masks = (boxes,x_keep,y_keep)
                
                # ========== V1 MODIFICATION: 使用新的 CLIP 输出 ==========
                # 原始: global_feat, local_feat = self.clip_head.image_encoder(...)
                # 新版: ho_tokens_out, cls_token, local_feat = self.clip_head.image_encoder(...)
                ho_tokens_out, cls_token, local_feat = self.clip_head.image_encoder(
                    image.decompose()[0][b_idx:b_idx + 1],
                    priors[b_idx] if self.args.use_prior else None,
                    ho_tokens,
                    mask, la_masks
                )
                # ======================================================

            # ========== V1 MODIFICATION: 归一化和门控融合 ==========
            # 归一化 ho_tokens
            ho_tokens_out = ho_tokens_out / ho_tokens_out.norm(dim=-1, keepdim=True)
            
            # 归一化 cls_token
            cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
            
            # 使用 SceneGate 融合
            if self.use_scene_gate:
                fused_tokens = self.scene_gate(ho_tokens_out, cls_token)
            else:
                fused_tokens = ho_tokens_out
            # ====================================================
            
            # 计算分数
            logits_text = fused_tokens @ text_features.T
            logits = logits_text.squeeze(0) * self.logit_scale_text.exp()

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
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

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)

        if self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1
        else:
            labels[x, targets['hoi'][y]] = 1
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])

        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = torch.cat(logits)
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
        torch.log(
            prior / (1 + torch.exp(-logits) - prior) + 1e-8
        ), labels, reduction='sum',
        alpha=self.alpha, gamma=self.gamma
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
            feat = feat[keep].view(-1,256)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
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
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        priors = torch.zeros((len(region_props),14,14), dtype=torch.float32, device=region_props[0]['boxes'].device)


        priors_dim = torch.zeros((len(region_props),14,14,max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] * (14 / scale_fct[b_idx][None,:])
            scores = props['scores']
            labels = props['labels']
            priors[b_idx] = len(boxes)


            boxes[:, 2:] += 0.5
            new_boxes = torch.round(boxes).long()

            for inb, nb in enumerate(new_boxes):
                x1_scaled, y1_scaled, x2_scaled, y2_scaled = nb
                priors[b_idx,y1_scaled:y2_scaled, x1_scaled:x2_scaled] = inb

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h,n)

            boxes = torch.cat([boxes,torch.tensor([[-1,-1,-1,-1.]]).to(boxes)],dim=0)
            labels = torch.cat([labels,torch.tensor([80]).to(boxes)],dim=0).long()
            scores = torch.cat([scores,torch.tensor([-1.]).to(boxes)],dim=0)

            object_embs = self.object_embedding[labels]

            sb = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            sb_feat = sb[priors[b_idx].long()]
            obj_feat = object_embs[priors[b_idx].long()]

            prior_feat = torch.cat([sb_feat,obj_feat],dim=-1)
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
        image_sizes_orig = torch.as_tensor([
            im.size()[-2:] for im in images_orig
            ], device=device)


        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)
        features, pos = self.detector.backbone(images_orig)
        src, mask = features[-1].decompose()
        hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])
        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()
        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'feats': hs[-1]}
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)

        priors = self.get_prior(region_props,image_sizes)

        images_clip = nested_tensor_from_tensor_list(images_clip)

        logits, prior, bh, bo, objects = self.compute_sim_scores(region_props,images_clip,priors)
        boxes = [r['boxes'] for r in region_props]

        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)

            loss_dict = dict(
                interaction_loss=interaction_loss
            )

            if self.args.local_rank == 0:
                wandb.log(loss_dict)

            return loss_dict

        if len(logits) == 0:
            print(targets)
            return None

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes):
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size,  in zip(
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
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model']) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    
    clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()
    clip_model = build_model(state_dict=clip_state_dict, args=args)

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
