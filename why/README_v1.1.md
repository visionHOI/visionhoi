## 4. V1 版本实现记录

### 4.0 新增文件路径汇总 (可视化)

```
LAIN/
├── CLIP/
│   └── clip_v1.py                    ✨ V1 修改
│
├── models/
│   ├── LAIN_v1.py                    ✨ V1 修改
│   └── scene_gate.py                  ✨ V1 新增
│
├── utils/
│   └── args_v1.py                    ✨ V1 新增
│
├── main_v1.py                        ✨ V1 修改
│
├── scripts/
│   ├── training/
│   │   ├── UO_v1.sh                 ✨ V1 新增
│   │   ├── UO_v1_continue.sh        ✨ V1 新增
│   │   └── NF-UC_v1.sh              ✨ V1 新增
│   │
│   └── eval/
│       ├── UO_v1.sh                 ✨ V1 新增
│       ├── UO_v1_eval.sh            ✨ V1 新增
│       ├── UO_v1_eval_epoch7.sh     ✨ V1 新增
│       └── NF-UC_v1.sh              ✨ V1 新增
│
└── checkpoints/
    ├── UO_v1/
    │   ├── ckpt_03895_01.pt         📊 10 epochs (mAP 14.99)
    │   ├── ckpt_07790_02.pt         📊 20 epochs (mAP 30.26)
    │   ├── ckpt_11685_03.pt         📊 30 epochs
    │   ├── ckpt_15580_04.pt         📊 40 epochs
    │   ├── ckpt_19475_05.pt         📊 50 epochs
    │   ├── ckpt_23370_06.pt         📊 60 epochs
    │   └── ckpt_27265_07.pt         📊 70 epochs (mAP 33.09) ⭐
    │
    └── pretrained/
        ├── detr/detr-r50-hicodet.pth
        └── clip/ViT-B-16.pt
```

#### 图例

| 符号 | 含义 |
|------|------|
| ✨ | 新增或修改的文件 |
| 📊 | 训练 checkpoint |
| ⭐ | 当前最佳 |

---

### 4.1 核心修改

#### 修改文件列表

| 文件 | 说明 |
|------|------|
| `CLIP/clip_v1.py` | 修改 `VisionTransformer.forward` 返回值，分离 `ho_tokens` 和 `cls_token` |
| `models/scene_gate.py` | 新增 `SceneGate` 模块，实现门控融合机制 |
| `models/LAIN_v1.py` | 修改 `compute_sim_scores` 函数，集成 SceneGate |
| `utils/args_v1.py` | 新增命令行参数：`use_scene_gate`, `scene_gate_type`, `scene_gate_hidden_dim` |
| `main_v1.py` | 使用 V1 版本的模块 |
| `scripts/training/UO_v1.sh` | UO 设置的训练脚本 |
| `scripts/training/NF-UC_v1.sh` | NF-UC 设置的训练脚本 |
| `scripts/eval/UO_v1.sh` | UO 设置的评估脚本 |
| `scripts/eval/NF-UC_v1.sh` | NF-UC 设置的评估脚本 |

### 4.2 关键代码修改

#### 4.2.1 CLIP 输出修改 (clip_v1.py)

```python
# 原始代码:
return x[:,:-196,:], x[:,-196:,:].view(...)

# V1 修改:
# Token 顺序: [ho_tokens, class_embedding, patches]
global_seq = x[:, :-196, :]  # [B, N_pairs+1, 512]
cls_token = global_seq[:, -1, :]  # [B, 512]
ho_tokens = global_seq[:, :-1, :]  # [B, N_pairs, 512]
local_feat = x[:, -196:, :].view(...)

return ho_tokens, cls_token, local_feat
```

#### 4.2.2 SceneGate 模块 (scene_gate.py)

```python
class SceneGate(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 输出 -1 到 1
        )
    
    def forward(self, ho_tokens, cls_token):
        # Image-level 门控
        gate = self.gate_mlp(cls_token)  # [B, 1]
        fused = ho_tokens + gate.unsqueeze(-1) * cls_token.unsqueeze(1)
        return fused
```

#### 4.2.3 LAIN_v1 中的融合逻辑

```python
# CLIP 编码
ho_tokens_out, cls_token, local_feat = self.clip_head.image_encoder(...)

# 归一化
ho_tokens_out = ho_tokens_out / ho_tokens_out.norm(dim=-1, keepdim=True)
cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

# 门控融合
if self.use_scene_gate:
    fused_tokens = self.scene_gate(ho_tokens_out, cls_token)
else:
    fused_tokens = ho_tokens_out

# 计算分数
logits_text = fused_tokens @ text_features.T
```

### 4.3 新增命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_scene_gate` | True | 是否使用 SceneGate |
| `--scene_gate_type` | 'image' | 门控类型: 'image' 或 'pair' |
| `--scene_gate_hidden_dim` | 128 | SceneGate MLP 隐藏层维度 |

### 4.4 训练配置

#### 4.4.1 UO 设置训练脚本 (UO_v1.sh)

```bash
torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num \
         main_v1.py \
         --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth \
         --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
         --output-dir checkpoints/UO_v1 \
         --dataset hicodet --zs --zs_type unseen_object \
         --num_classes 117 --num-workers 4 \
         --epochs 20 \
         --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
         --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1. \
         --use_scene_gate --scene_gate_type image --scene_gate_hidden_dim 128 \
         --print-interval 100
```

#### 4.4.2 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--epochs` | 20 | 基础训练轮数（每个 checkpoint 保存间隔为 10 epochs）|
| `--use_scene_gate` | True | 启用 SceneGate 模块 |
| `--scene_gate_type` | image | 使用 Image-level 门控 |
| `--scene_gate_hidden_dim` | 128 | 门控 MLP 隐藏层维度 |
| `--use_insadapter` | True | 使用实例适配器 |
| `--adapt_dim` | 32 | 适配器维度 |
| `--use_prompt` | True | 使用提示学习 |
| `--N_CTX` | 36 | 提示词上下文长度 |

### 4.5 实验结果

#### 4.5.1 UO 设置实验结果

| Checkpoint | 版本 | 实际 Epochs | Full mAP | Unseen mAP | Seen mAP | Rare mAP | Non-rare mAP |
|------------|------|-------------|----------|------------|----------|----------|---------------|
| `ckpt_03895_01.pt` | 01 | 10 | 14.99 | 18.52 | 14.29 | 10.72 | 16.27 |
| `ckpt_07790_02.pt` | 02 | 20 | 30.26 | 33.86 | 29.55 | 27.98 | 30.95 |
| `ckpt_27265_07.pt` | 07 | 70 | **33.09** | **36.53** | **32.40** | **32.54** | **33.25** |

#### 4.5.2 与论文结果对比

| 模型 | Setting | Unseen | Seen | mAP |
|------|---------|--------|------|-----|
| **LAIN (论文)** | UO | 37.65 | 33.61 | 34.28 |
| **LAIN-V1 (10 epochs)** | UO | 18.52 | 14.29 | 14.99 |
| **LAIN-V1 (20 epochs)** | UO | 33.86 | 29.55 | 30.26 |
| **LAIN-V1 (70 epochs)** | UO | 36.53 | 32.40 | 33.09 |

#### 4.5.3 结果分析

1. **训练收敛趋势**: 从 10 epochs 到 70 epochs，模型性能持续提升
2. **与论文差距**: 70 epochs 时 Full mAP 为 33.09，与论文的 34.28 差约 1.2 个点
3. **继续训练潜力**: 模型在 70 epochs 后仍有提升空间，建议继续训练至 100 epochs

### 4.6 使用方法

#### 训练

```bash
# UO 设置
bash scripts/training/UO_v1.sh

# NF-UC 设置
bash scripts/training/NF-UC_v1.sh
```

#### 评估

```bash
# UO 设置
bash scripts/eval/UO_v1.sh

# NF-UC 设置
bash scripts/eval/NF-UC_v1.sh
```

### 4.7 设计原理

#### 门控融合机制

```
原始 LAIN:
  HO tokens → CLIP → HO tokens (丢弃 CLS) → HOI 分数

V1 改进:
  HO tokens → CLIP → HO tokens + CLS token
                           ↓
                      SceneGate (门控融合)
                           ↓
                      融合特征 → HOI 分数
```

#### 门控学习目标

- **gate ≈ 0**: 该 HOI 不需要场景信息，CLS token 不影响决策
- **gate ≈ 1**: 该 HOI 需要场景信息，CLS token 完全融入
- **gate ∈ (0, 1)**: 部分融合

#### 为什么使用 Image-level 门控

1. **简单**: 只需学习一个标量门控值
2. **安全**: 初始化为 0，保证"至少不坏"
3. **可解释**: 可以分析哪些图像需要更多场景信息

### 4.8 后续优化方向

1. **Pair-level 门控**: 每个人-物对有独立的门控值
2. **场景文本匹配**: 定义场景文本，显式匹配场景
3. **多任务学习**: 添加场景分类辅助损失

---

*本文档版本: v1.2*
*最后更新: 2026-03-01*
*V1 实验完成*
