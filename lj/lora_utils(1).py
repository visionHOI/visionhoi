# models/lora_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------- 1. 低秩线性层（LoRA） ----------------------
class LowRankLinear(nn.Module):
    """
    低秩线性层：W = W_pretrain + A * B * alpha / rank
    - W_pretrain：预训练权重（冻结）
    - A/B：低秩矩阵（可训练）
    - alpha：缩放因子，平衡低秩层贡献
    """

    def __init__(self, in_dim, out_dim, rank=8, alpha=16, bias=True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # 预训练权重（冻结，不参与梯度更新）
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(out_dim)) if bias else None

        # 低秩矩阵（可训练，初始化遵循LoRA论文）
        self.A = nn.Parameter(torch.randn(in_dim, rank) / math.sqrt(rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        # 原预训练权重计算
        out = F.linear(x, self.weight, self.bias)
        # 低秩增量计算（带缩放，仅训练这部分）
        low_rank_out = (x @ self.A @ self.B) * self.scale
        return out + low_rank_out


# ---------------------- 2. 低秩MLP（替换原有MLP） ----------------------
class LowRankMLP(nn.Module):
    """低秩MLP：所有线性层替换为低秩层，保持原MLP激活逻辑"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, rank=8, alpha=16):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                # 输入层：低秩线性层
                layers.append(LowRankLinear(input_dim, hidden_dim, rank, alpha))
            elif i == num_layers - 1:
                # 输出层：低秩线性层
                layers.append(LowRankLinear(hidden_dim, output_dim, rank, alpha))
            else:
                # 隐藏层：低秩线性层
                layers.append(LowRankLinear(hidden_dim, hidden_dim, rank, alpha))
            # 保持和原MLP一致的ReLU激活
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


# ---------------------- 3. 语义适配器（双模式：简单/复杂） ----------------------
class TextSemanticAdapter(nn.Module):
    """
    文本侧语义适配器（支持两种训练模式）
    - 简单模式：基础残差适配器（LayerNorm+降维+GELU+升维）
    - 复杂模式：多粒度特征拆分+门控融合+低秩残差增强
    """

    def __init__(self,
                 dim=768,  # 输入维度（CLIP文本特征维度）
                 bottleneck=64,  # 降维瓶颈维度
                 rank=8,  # 低秩分解的秩（仅复杂模式用）
                 alpha=16,  # 低秩缩放因子（仅复杂模式用）
                 adapter_mode='simple'  # 适配器模式：simple/complex
                 ):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        self.adapter_mode = adapter_mode  # 模式开关

        # ====================== 模式1：简单适配器（基础版） ======================
        """
        简单模式逻辑：
        原始特征 → LayerNorm → 降维 → GELU → 升维 → 残差连接
        特点：参数量少、训练快，适合快速适配/小数据集
        """
        self.simple_norm = nn.LayerNorm(dim)
        self.simple_down = nn.Linear(dim, bottleneck, bias=False)
        self.simple_act = nn.GELU()
        self.simple_up = nn.Linear(bottleneck, dim, bias=False)
        # 初始化：升维层权重置0，保证初始为恒等变换
        nn.init.zeros_(self.simple_up.weight)

        # ====================== 模式2：复杂适配器（多粒度+门控+低秩） ======================
        """
        复杂模式逻辑：
        1. 多粒度特征拆分：单词级（动作/物体）、短语级、句子级
        2. 多粒度特征变换：各自降维+低秩增强
        3. 门控融合：自适应加权三个粒度的特征
        4. 低秩残差增强：升维回768维，叠加到原始embedding
        特点：语义表达更强，适合大数据集/高精度需求
        """
        if self.adapter_mode == 'complex':
            # 3.1 多粒度特征拆分（基于CLIP文本token的位置拆分）
            # 假设文本token结构：<CLS> + 动作单词 + 物体单词 + 短语 + 句子 + <EOS>
            # 拆分映射：单词级（前10个token）、短语级（中间10个）、句子级（剩余）
            self.register_buffer('word_mask', torch.zeros(dim))
            self.register_buffer('phrase_mask', torch.zeros(dim))
            self.register_buffer('sentence_mask', torch.zeros(dim))
            # 手动初始化掩码（模拟不同粒度的特征拆分，可根据实际token调整）
            self.word_mask[:256] = 1.0  # 单词级：前256维（动作/物体核心语义）
            self.phrase_mask[256:512] = 1.0  # 短语级：中间256维
            self.sentence_mask[512:] = 1.0  # 句子级：后256维

            # 3.2 多粒度特征变换（降维+低秩增强）
            # 单词级变换
            self.word_down = nn.Linear(dim, bottleneck)
            self.word_lowrank = LowRankLinear(bottleneck, bottleneck, rank, alpha)
            # 短语级变换
            self.phrase_down = nn.Linear(dim, bottleneck)
            self.phrase_lowrank = LowRankLinear(bottleneck, bottleneck, rank, alpha)
            # 句子级变换
            self.sentence_down = nn.Linear(dim, bottleneck)
            self.sentence_lowrank = LowRankLinear(bottleneck, bottleneck, rank, alpha)

            # 3.3 门控融合（自适应加权）
            self.gate = nn.Sequential(
                nn.Linear(bottleneck * 3, 3),  # 输入：三个粒度特征拼接
                nn.Softmax(dim=-1)  # 输出：权重（和为1）
            )

            # 3.4 低秩残差增强（升维+低秩）
            self.complex_norm = nn.LayerNorm(dim)
            self.fusion_down = nn.Linear(bottleneck, bottleneck * 2)
            self.fusion_act = nn.GELU()
            self.fusion_up = LowRankLinear(bottleneck * 2, dim, rank, alpha)
            # 初始化：低秩层B矩阵置0，保证初始无影响
            nn.init.zeros_(self.fusion_up.B)

    def forward(self, x):
        """
        前向传播：根据模式选择不同逻辑
        x: 输入文本特征，shape=[batch, seq_len, dim] 或 [batch, dim]
        """
        residual = x  # 原始特征残差
        if len(x.shape) == 2:  # [batch, dim] → 扩展为3维，统一处理
            x = x.unsqueeze(1)

        # ====================== 模式1：简单适配器 ======================
        if self.adapter_mode == 'simple':
            x_norm = self.simple_norm(x)
            x_down = self.simple_down(x_norm)
            x_act = self.simple_act(x_down)
            x_up = self.simple_up(x_act)
            output = residual + x_up  # 残差连接

        # ====================== 模式2：复杂适配器 ======================
        elif self.adapter_mode == 'complex':
            # 步骤1：多粒度特征拆分（基于掩码）
            word_feat = x * self.word_mask.unsqueeze(0).unsqueeze(0)  # 单词级特征
            phrase_feat = x * self.phrase_mask.unsqueeze(0).unsqueeze(0)  # 短语级特征
            sentence_feat = x * self.sentence_mask.unsqueeze(0).unsqueeze(0)  # 句子级特征

            # 步骤2：多粒度特征变换（降维+低秩增强）
            word_feat = self.word_lowrank(self.word_down(word_feat))  # 单词级：降维+低秩
            phrase_feat = self.phrase_lowrank(self.phrase_down(phrase_feat))  # 短语级
            sentence_feat = self.sentence_lowrank(self.sentence_down(sentence_feat))  # 句子级

            # 步骤3：门控融合（自适应加权）
            fusion_input = torch.cat([word_feat, phrase_feat, sentence_feat], dim=-1)  # [B, L, 3*bottleneck]
            gate_weight = self.gate(fusion_input)  # [B, L, 3] → 三个粒度的权重
            # 加权融合：weight[:,0]→单词级，weight[:,1]→短语级，weight[:,2]→句子级
            fused_feat = (word_feat * gate_weight[..., 0:1] +
                          phrase_feat * gate_weight[..., 1:2] +
                          sentence_feat * gate_weight[..., 2:3])

            # 步骤4：低秩残差增强（升维+残差）
            x_norm = self.complex_norm(x)
            fusion_down = self.fusion_act(self.fusion_down(fused_feat))
            fusion_up = self.fusion_up(fusion_down)  # 低秩升维回768维
            output = residual + fusion_up  # 叠加到原始特征

        # 维度还原（如果输入是2维）
        if len(residual.shape) == 2:
            output = output.squeeze(1)

        return output