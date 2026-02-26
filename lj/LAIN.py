"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import sys
import math
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torchvision.ops.boxes import batched_nms, box_iou

import numpy as np
import wandb

# 自定义工具类导入
from utils.hico_list import hico_verbs_sentence
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices
from utils.postprocessor import PostProcess
from utils.ops import binary_focal_loss_with_logits
from utils import hico_text_label


# ====================== 低秩/文本适配器模块 ======================
class LowRankLinear(nn.Module):
    """低秩线性层（LoRA）：仅训练A/B矩阵，冻结主干权重"""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: int = 16, bias: bool = True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # 预训练权重（冻结）
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        # 低秩矩阵（可训练）
        self.A = nn.Parameter(torch.randn(in_dim, rank) / math.sqrt(rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, self.bias)
        low_rank_out = (x @ self.A @ self.B) * self.scale
        return out + low_rank_out


class LowRankMLP(nn.Module):
    """低秩MLP：替换原始MLP，用于query_proj/priors_downproj"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, rank: int = 8,
                 alpha: int = 16):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(LowRankLinear(input_dim, hidden_dim, rank, alpha))
            elif i == num_layers - 1:
                layers.append(LowRankLinear(hidden_dim, output_dim, rank, alpha))
            else:
                layers.append(LowRankLinear(hidden_dim, hidden_dim, rank, alpha))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TextSemanticAdapter(nn.Module):
    """文本侧语义适配器（动态维度版，适配ViT-B/16的512维）"""

    def __init__(self, bottleneck: int = 64, input_dim: Optional[int] = None):
        super().__init__()
        # 若未指定input_dim，forward时自动适配输入维度
        self.fixed_dim = input_dim
        self.bottleneck = bottleneck
        self.norm: Optional[nn.LayerNorm] = None  # 延迟初始化，适配输入维度
        self.down: Optional[nn.Linear] = None
        self.up: Optional[nn.Linear] = None
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # 首次forward时根据输入动态初始化层
        if self.norm is None:
            input_dim = self.fixed_dim if self.fixed_dim is not None else x.shape[-1]
            self.norm = nn.LayerNorm(input_dim).to(x.device)
            self.down = nn.Linear(input_dim, self.bottleneck).to(x.device)
            self.up = nn.Linear(self.bottleneck, input_dim).to(x.device)
            # 初始化：升维层权重置0，保证初始为恒等变换
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return residual + x


# ====================== CLIP和DETR相关导入 ======================
try:
    from CLIP.clip import build_model
    from CLIP.customCLIP import CustomCLIP, tokenize
except ImportError as e:
    raise ImportError(f"CLIP module import failed: {e}")

# 添加DETR路径并导入相关模块
sys.path.insert(0, 'detr')
try:
    from detr.models.backbone import build_backbone
    from detr.models.transformer import build_transformer
    from detr.models.detr import DETR
    from detr.util import box_ops
    from detr.util.misc import nested_tensor_from_tensor_list
finally:
    sys.path.pop(0)  # 恢复路径


# ====================== 基础MLP模块 ======================
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ====================== 核心LAIN模型 ======================
class LAIN(nn.Module):
    def __init__(self,
                 args,
                 detector: nn.Module,
                 postprocessor: nn.Module,
                 model: nn.Module,
                 object_embedding: Tensor,
                 human_idx: int,
                 num_classes: int,
                 alpha: float = 0.5,
                 gamma: float = 2.0,
                 box_score_thresh: float = 0.2,
                 fg_iou_thresh: float = 0.5,
                 min_instances: int = 3,
                 max_instances: int = 15,
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
            object_n_verb_to_interaction, dtype=np.float32
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
        self.tp: Optional[Tensor] = None  # 存储文本特征，避免重复计算
        self.reserve_indices = reserve_indices

        self.priors_initial_dim = self.visual_output_dim + 5
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ====================== 低秩MLP替换原始MLP ======================
        if getattr(args, 'use_lora', False):
            self.priors_downproj = LowRankMLP(
                self.priors_initial_dim, 128, args.adapt_dim, 3,
                rank=args.lora_rank, alpha=args.lora_alpha
            )
            self.query_proj = LowRankMLP(
                512, 128, 768, 2,
                rank=args.lora_rank, alpha=args.lora_alpha
            )
        else:
            self.priors_downproj = MLP(self.priors_initial_dim, 128, args.adapt_dim, 3)
            self.query_proj = MLP(512, 128, 768, 2)

        # ====================== 文本语义适配器初始化 ======================
        self.use_text_adapter = getattr(args, 'use_text_adapter', False)
        if self.use_text_adapter:
            # 显式指定input_dim=512（适配ViT-B/16）
            self.text_adapter = TextSemanticAdapter(
                bottleneck=args.text_adapter_dim,
                input_dim=512  # 关键：强制适配ViT-B/16的512维
            )

    def compute_prior_scores(self,
                             x: Tensor,
                             y: Tensor,
                             scores: Tensor,
                             object_class: Tensor
                             ) -> Tensor:
        """计算先验分数"""
        batch_size = len(x)
        prior_h = torch.zeros(batch_size, self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # 批量处理目标类别索引，提升效率
        target_cls_idx = []
        for obj in object_class[y]:
            target_cls_idx.extend(self.object_class_to_target_class[obj.item()])
        pair_idx = []
        flat_target_idx = []
        for i, tar in enumerate(target_cls_idx):
            pair_idx.extend([i // len(tar)] * len(tar))  # 修正pair_idx计算逻辑
            flat_target_idx.extend(tar)

        if pair_idx and flat_target_idx:
            prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
            prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def compute_sim_scores(self,
                           region_props: List[Dict],
                           image,
                           priors=None) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """计算相似度分数
        Returns:
            all_logits: 每个批次的相似度logits
            prior_collated: 每个批次的先验分数
            boxes_h_collated: 每个批次的人类框索引
            boxes_o_collated: 每个批次的物体框索引
            object_class_collated: 每个批次的物体类别
        """
        device = image.tensors.device
        boxes_h_collated = []
        boxes_o_collated = []
        prior_collated = []
        object_class_collated = []
        all_logits = []

        # 获取文本特征（仅计算一次）
        text_features = self._get_text_features()

        # 处理每一批的HO tokens
        for b_idx, props in enumerate(region_props):
            batch_logits, batch_prior, batch_h, batch_o, batch_obj = self._process_single_batch(
                b_idx, props, text_features, device
            )
            all_logits.append(batch_logits)
            prior_collated.append(batch_prior)
            boxes_h_collated.append(batch_h)
            boxes_o_collated.append(batch_o)
            object_class_collated.append(batch_obj)

        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated

    def _get_text_features(self) -> Optional[Tensor]:
        """获取文本特征（带缓存和适配器）"""
        if not self.args.use_prompt:
            return None

        # 训练模式：每次重新计算；推理模式：缓存复用
        if self.training or self.tp is None:
            prompts = self.clip_head.prompt_learner()
            text_features = self.clip_head.text_encoder(prompts, self.clip_head.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            if not self.training:
                self.tp = text_features
        else:
            text_features = self.tp

        # 应用文本语义适配器
        if self.use_text_adapter and text_features is not None:
            text_features = self.text_adapter(text_features)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _process_single_batch(self,
                              b_idx: int,
                              props: Dict,
                              text_features: Optional[Tensor],
                              device: torch.device) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """处理单个批次的HO对计算"""
        boxes = props['boxes']
        scores = props['scores']
        labels = props['labels']
        feats = props['feat']

        # 筛选人类实例
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        n_total = len(boxes)

        # 无有效人类/物体对时返回空张量
        if n_h == 0 or n_total <= 1:
            empty_logits = torch.zeros(0, self.num_classes, device=device)
            empty_prior = torch.zeros(2, 0, self.num_classes, device=device)
            empty_idx = torch.zeros(0, device=device, dtype=torch.int64)
            return empty_logits, empty_prior, empty_idx, empty_idx, empty_idx

        # 重新排列：人类在前，物体在后
        h_idx = torch.nonzero(is_human).squeeze(1)
        o_idx = torch.nonzero(~is_human).squeeze(1)
        perm = torch.cat([h_idx, o_idx])
        boxes = boxes[perm]
        scores = scores[perm]
        labels = labels[perm]
        feats = feats[perm]

        # 生成人类-物体对（排除自对）
        x, y = torch.meshgrid(
            torch.arange(n_total, device=device),
            torch.arange(n_total, device=device),
            indexing='ij'
        )
        valid_mask = torch.logical_and(x != y, x < n_h)
        x_keep, y_keep = torch.nonzero(valid_mask).unbind(1)

        if len(x_keep) == 0:
            empty_logits = torch.zeros(0, self.num_classes, device=device)
            empty_prior = torch.zeros(2, 0, self.num_classes, device=device)
            empty_idx = torch.zeros(0, device=device, dtype=torch.int64)
            print(f"[Batch {b_idx}] No valid HO pairs, skip")
            return empty_logits, empty_prior, empty_idx, empty_idx, empty_idx

        # 计算HO特征和文本相似度
        logits = self._compute_ho_similarity(feats, x_keep, y_keep, text_features, device)

        # 计算先验分数
        prior = self.compute_prior_scores(x_keep, y_keep, scores, labels)

        return logits, prior, x_keep, y_keep, labels[y_keep]

    def _compute_ho_similarity(self,
                               feats: Tensor,
                               x_keep: Tensor,
                               y_keep: Tensor,
                               text_features: Optional[Tensor],
                               device: torch.device) -> Tensor:
        """计算HO对与文本的相似度分数"""
        if self.args.use_hotoken and text_features is not None:
            # 拼接人类和物体特征
            ho_feat = torch.cat([feats[x_keep], feats[y_keep]], dim=-1)
            ho_tokens = self.query_proj(ho_feat)

            # 构建注意力掩码
            num_tokens = len(x_keep) + 1
            mask = torch.zeros((num_tokens + 196, num_tokens + 196), dtype=torch.bool, device=device)
            mask[:num_tokens, :num_tokens] = ~torch.eye(num_tokens, dtype=torch.bool, device=device)
            mask[-197:, :-197] = True

            # 提取视觉特征
            la_masks = (feats, x_keep, y_keep)
            global_feat, _ = self.clip_head.image_encoder(
                feats.unsqueeze(0),  # 适配batch维度
                None,  # priors在单批次处理中单独传入
                ho_tokens,
                mask,
                la_masks
            )

            # 归一化并计算文本相似度
            global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
            global_feat = global_feat[:, :-1]  # 移除padding token
            logits_text = global_feat @ text_features.T
            logits = logits_text.squeeze(0) * self.logit_scale_text.exp()
        else:
            # 无HO token时返回空logits
            logits = torch.zeros(len(x_keep), self.num_classes, device=device)

        return logits

    def recover_boxes(self, boxes: Tensor, size: Tuple[int, int]) -> Tensor:
        """将cxcywh格式的框转换为xyxy并恢复到原始尺寸"""
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h]).to(boxes.device)
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self,
                                    boxes_h: Tensor,
                                    boxes_o: Tensor,
                                    targets: Dict) -> Tensor:
        """将预测的HO对与真实标签关联"""
        n_pairs = boxes_h.shape[0]
        labels = torch.zeros(n_pairs, self.num_classes, device=boxes_h.device)

        # 恢复真实框到原始尺寸
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        # 计算IOU并匹配正样本（双框IOU都需满足阈值）
        iou_h = box_iou(boxes_h, gt_bx_h)
        iou_o = box_iou(boxes_o, gt_bx_o)
        min_iou = torch.min(iou_h, iou_o)
        match_idx = torch.nonzero(min_iou >= self.fg_iou_thresh).unbind(1)

        if len(match_idx) == 2:
            x, y = match_idx
            # 根据数据集类型分配标签
            if self.num_classes in [117, 24, 407]:
                labels[x, targets['labels'][y]] = 1.0
            else:
                labels[x, targets['hoi'][y]] = 1.0

        return labels

    def compute_interaction_loss(self,
                                 boxes: List[Tensor],
                                 bh: List[Tensor],
                                 bo: List[Tensor],
                                 logits: List[Tensor],
                                 prior: List[Tensor],
                                 targets: List[Dict]) -> Tensor:
        """计算交互损失（Focal Loss）"""
        # 收集所有批次的真实标签
        labels_list = []
        for bx, h_idx, o_idx, target in zip(boxes, bh, bo, targets):
            bx_h = self.recover_boxes(bx[h_idx], target['size'])
            bx_o = self.recover_boxes(bx[o_idx], target['size'])
            labels_list.append(self.associate_with_ground_truth(bx_h, bx_o, target))

        if not labels_list:
            return torch.tensor(0.0, device=boxes[0].device)

        labels = torch.cat(labels_list)
        prior = torch.cat(prior, dim=1).prod(0)

        # 只处理非零先验分数的样本
        valid_mask = prior > 1e-8  # 宽松阈值避免数值误差
        x, y = torch.nonzero(valid_mask).unbind(1)

        if len(x) == 0:
            return torch.tensor(0.0, device=labels.device)

        # 筛选有效样本
        logits = torch.cat(logits)[x, y]
        prior = prior[x, y]
        labels = labels[x, y]

        # 分布式训练下的正样本数归一化
        n_pos = torch.sum(labels > 0.5).float()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
            dist.all_reduce(n_pos)
            n_pos = n_pos / dist.get_world_size()

        # 计算Focal Loss（带数值稳定）
        logits_input = torch.log(
            prior / (1 + torch.exp(-logits) - prior + 1e-8) + 1e-8
        )
        loss = binary_focal_loss_with_logits(
            logits_input, labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )

        return loss / max(n_pos.item(), 1e-6)  # 防止除零

    def prepare_region_proposals(self, results: Dict) -> List[Dict]:
        """准备区域提议（过滤、NMS、数量限制）"""
        region_props = []
        for res in results:
            # 分离计算图，节省内存
            scores = res['scores'].detach()
            labels = res['labels'].detach()
            boxes = res['boxes'].detach()
            feats = res['feats'].detach()

            # NMS过滤重复框
            keep = batched_nms(boxes, scores, labels, 0.5)
            scores = scores[keep].view(-1)
            labels = labels[keep].view(-1)
            boxes = boxes[keep].view(-1, 4)
            feats = feats[keep].view(-1, feats.shape[-1])

            # 分数阈值过滤
            score_mask = scores >= self.box_score_thresh
            keep = torch.nonzero(score_mask).squeeze(1)
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            feats = feats[keep]

            # 分离人类和物体实例
            is_human = labels == self.human_idx
            hum_idx = torch.nonzero(is_human).squeeze(1)
            obj_idx = torch.nonzero(~is_human).squeeze(1)

            # 限制实例数量（保证最少/最多实例数）
            hum_keep = self._limit_instances(hum_idx, scores, self.min_instances, self.max_instances)
            obj_keep = self._limit_instances(obj_idx, scores, self.min_instances, self.max_instances)

            # 合并并去重
            keep = torch.cat([hum_keep, obj_keep]).unique()

            region_props.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'labels': labels[keep],
                'feat': feats[keep]
            })

        return region_props

    def _limit_instances(self,
                         indices: Tensor,
                         scores: Tensor,
                         min_inst: int,
                         max_inst: int) -> Tensor:
        """限制实例数量（按分数排序）"""
        n_inst = len(indices)
        if n_inst == 0:
            return indices
        # 按分数降序排序
        sorted_idx = indices[scores[indices].argsort(descending=True)]
        # 限制数量
        if n_inst < min_inst:
            return sorted_idx[:min_inst]
        elif n_inst > max_inst:
            return sorted_idx[:max_inst]
        return sorted_idx

    def get_prior(self, region_props: List[Dict], image_size: Tensor) -> Tensor:
        """生成先验特征"""
        if not region_props:
            return torch.empty(0)

        max_feat_dim = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        device = region_props[0]['boxes'].device
        batch_size = len(region_props)

        # 初始化先验张量
        priors = torch.zeros((batch_size, 14, 14), dtype=torch.float32, device=device)
        priors_dim = torch.zeros((batch_size, 14, 14, max_feat_dim), dtype=torch.float32, device=device)

        # 计算缩放因子
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            n_boxes = len(boxes)

            # 记录框数量
            priors[b_idx] = n_boxes
            if n_boxes == 0:
                continue

            # 缩放框到14x14网格
            boxes_scaled = boxes * (14.0 / scale_fct[b_idx][None, :])
            boxes_scaled[:, 2:] += 0.5  # 中心对齐
            boxes_rounded = torch.round(boxes_scaled).long()

            # 填充先验矩阵（框索引）
            for inb, nb in enumerate(boxes_rounded):
                x1, y1, x2, y2 = nb.clamp(0, 13)
                priors[b_idx, y1:y2, x1:x2] = inb

            # 补充无效框用于维度对齐
            boxes_padded = torch.cat([boxes, torch.tensor([[-1, -1, -1, -1.]]).to(device)], dim=0)
            labels_padded = torch.cat([labels, torch.tensor([80]).to(device)], dim=0).long()
            scores_padded = torch.cat([scores, torch.tensor([-1.]).to(device)], dim=0)

            # 获取物体嵌入
            object_embs = self.object_embedding[labels_padded]

            # 拼接分数、框特征和物体嵌入
            sb = torch.cat((scores_padded.unsqueeze(-1), boxes_padded), dim=-1)
            sb_feat = sb[priors[b_idx].long()]
            obj_feat = object_embs[priors[b_idx].long()]
            prior_feat = torch.cat([sb_feat, obj_feat], dim=-1)

            priors_dim[b_idx] = prior_feat

        # 通过低秩MLP降维
        priors = self.priors_downproj(priors_dim)

        return priors

    def forward(self,
                images: List[Tensor],
                targets: Optional[List[Dict]] = None
                ) -> List[Dict]:
        """前向传播
        Args:
            images: 列表，每个元素为tuple(原始图像, CLIP输入图像)
            targets: 训练时的真实标签
        Returns:
            训练模式：loss_dict；推理模式：检测结果
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        batch_size = len(images)
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]
        device = images_clip[0].device

        # 处理图像尺寸
        image_sizes = torch.as_tensor([im.size()[-2:] for im in images_clip], device=device)
        image_sizes_orig = torch.as_tensor([im.size()[-2:] for im in images_orig], device=device)

        # 1. 提取检测特征（DETR backbone + transformer）
        images_orig_nt = nested_tensor_from_tensor_list(images_orig)
        features, pos = self.detector.backbone(images_orig_nt)
        src, mask = features[-1].decompose()
        hs, detr_memory = self.detector.transformer(
            self.detector.input_proj(src), mask,
            self.detector.query_embed.weight, pos[-1]
        )

        # 2. 检测头预测
        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()

        # VCoco数据集类别过滤
        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'

        # 3. 后处理检测结果
        det_results = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'feats': hs[-1]
        }
        det_results = self.postprocessor(det_results, image_sizes)
        region_props = self.prepare_region_proposals(det_results)

        # 4. 生成先验特征
        priors = self.get_prior(region_props, image_sizes)

        # 5. 处理CLIP图像并计算相似度分数
        images_clip_nt = nested_tensor_from_tensor_list(images_clip)
        logits, prior, bh, bo, objects = self.compute_sim_scores(region_props, images_clip_nt, priors)
        boxes = [r['boxes'] for r in region_props]

        # 6. 训练模式：计算损失
        if self.training:
            # 过滤空批次
            valid_idx = [i for i, lg in enumerate(logits) if lg.numel() > 0]
            if len(valid_idx) == 0:
                loss_dict = {'interaction_loss': torch.tensor(0.0, device=device)}
                if self.args.local_rank == 0:
                    wandb.log(loss_dict)
                return loss_dict

            # 筛选有效数据
            logits = [logits[i] for i in valid_idx]
            prior = [prior[i] for i in valid_idx]
            bh = [bh[i] for i in valid_idx]
            bo = [bo[i] for i in valid_idx]
            boxes = [boxes[i] for i in valid_idx]
            targets = [targets[i] for i in valid_idx]
            objects = [objects[i] for i in valid_idx]

            # 计算交互损失
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)

            loss_dict = {'interaction_loss': interaction_loss}
            if self.args.local_rank == 0:
                wandb.log(loss_dict)
            return loss_dict

        # 7. 推理模式：后处理生成检测结果
        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

    def postprocessing(self,
                       boxes: List[Tensor],
                       bh: List[Tensor],
                       bo: List[Tensor],
                       logits: List[Tensor],
                       prior: List[Tensor],
                       objects: List[Tensor],
                       image_sizes: Tensor) -> List[Dict]:
        """推理阶段的后处理"""
        detections = []
        n_list = [len(b) for b in bh]
        logits = torch.cat(logits).split(n_list)

        for bx, h_idx, o_idx, lg, pr, obj, size in zip(
                boxes, bh, bo, logits, prior, objects, image_sizes
        ):
            pr_prod = pr.prod(0)
            # 筛选非零先验分数的HO对
            valid_mask = pr_prod > 1e-8
            x, y = torch.nonzero(valid_mask).unbind(1)

            if len(x) == 0:
                # 无有效HO对
                detections.append({
                    'boxes': bx,
                    'pairing': torch.empty((2, 0), dtype=torch.int64),
                    'scores': torch.empty(0, device=bx.device),
                    'labels': torch.empty(0, dtype=torch.int64),
                    'objects': torch.empty(0, dtype=torch.int64),
                    'size': size
                })
                continue

            # 计算最终分数（sigmoid + 先验分数）
            scores = torch.sigmoid(lg[x, y]) * pr_prod[x, y]

            detections.append({
                'boxes': bx,
                'pairing': torch.stack([h_idx[x], o_idx[x]]),
                'scores': scores,
                'labels': y,
                'objects': obj[x],
                'size': size
            })

        return detections


# ====================== 辅助函数 ======================
@torch.no_grad()
def get_obj_text_emb(args, clip_model, obj_class_names: List[str]) -> Tensor:
    """获取物体的文本嵌入（批量处理）"""
    # 批量tokenize，提升效率
    obj_text_inputs = tokenize(obj_class_names)
    obj_text_embedding = clip_model.encode_text(obj_text_inputs)
    object_embedding = obj_text_embedding / obj_text_embedding.norm(dim=-1, keepdim=True)
    return object_embedding


def build_detector(args,
                   class_corr: List[list],
                   object_n_verb_to_interaction: List[list],
                   clip_model_path: str) -> LAIN:
    """构建LAIN检测器
    Args:
        args: 配置参数
        class_corr: 物体类别到目标类别的映射
        object_n_verb_to_interaction: 物体-动词到交互的映射
        clip_model_path: CLIP模型权重路径
    Returns:
        LAIN检测器实例
    """
    # 确定检测器类别数
    num_classes = 80
    if args.dataset == 'vcoco' and 'e632da11' in args.pretrained:
        num_classes = 91

    # 1. 构建DETR检测器
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    detr = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    # 加载预训练权重
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(checkpoint['model'])
        else:
            detr.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    else:
        if dist.get_rank() == 0:
            print(f"Warning: Pretrained detector weights not found at {args.pretrained}")

    # 2. 构建CLIP模型
    clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()
    clip_model = build_model(
        state_dict=clip_state_dict,
        use_adapter=args.use_insadapter,
        adapter_pos=args.adapter_pos,
        args=args
    )

    # 3. 加载类别名称
    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError(f"Unsupported num_classes: {args.num_classes}")

    # 4. 构建CustomCLIP
    model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)

    # 5. 获取物体文本嵌入
    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]
    object_embedding = get_obj_text_emb(args, clip_model=clip_model, obj_class_names=obj_class_names)
    object_embedding = object_embedding.clone().detach()

    # 6. 构建后处理器
    postprocessors = {'bbox': PostProcess()}

    # 7. 构建LAIN模型
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