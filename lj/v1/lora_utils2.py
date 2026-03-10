import torch
import torch.nn as nn
import math


class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=16, bias=False):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)

        # 兼容PyTorch版本：用relu替代gelu
        nn.init.kaiming_normal_(self.A.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.B.weight, mode='fan_in', nonlinearity='relu')
        self.B.weight.data *= 0.01

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


class TextSemanticAdapter(nn.Module):
    def __init__(self, dim=512, bottleneck=64, rank=4, alpha=16, adapter_mode='complex'):
        super().__init__()
        self.dim = dim
        self.adapter_mode = adapter_mode

        # 基础通路（匹配指令参数：text_adapter_dim=64）
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.lowrank = LowRankLinear(bottleneck, bottleneck, rank, alpha)
        self.up = nn.Linear(bottleneck, dim, bias=False)

        # 兼容PyTorch版本：用relu替代gelu
        nn.init.kaiming_normal_(self.up.weight, mode='fan_in', nonlinearity='relu')
        self.up.weight.data *= 0.01

    # 核心修复：支持多参数传入（*args忽略多余参数）
    def forward(self, x, *args, **kwargs):
        # args会接收ho_feat_global等多余参数，直接忽略不处理
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.lowrank(x)
        x = self.up(x)
        return residual + x * 0.1