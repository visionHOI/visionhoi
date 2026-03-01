import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneGate(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        last_layer = self.gate_mlp[-2]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
    
    def forward(self, ho_tokens: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Image-level 门控融合
        
        Args:
            ho_tokens: [B, N_pairs, D] 人-物对特征
            cls_token: [B, D] 全局特征 (class_embedding)
        
        Returns:
            fused_tokens: [B, N_pairs, D] 融合后的特征
        """
        gate = self.gate_mlp(cls_token)
        
        cls_expanded = cls_token.unsqueeze(1)
        
        fused_tokens = ho_tokens + gate.unsqueeze(-1) * cls_expanded
        
        return fused_tokens


class SceneGatePair(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.pair_gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.pair_gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        last_layer = self.pair_gate_mlp[-2]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
    
    def forward(self, ho_tokens: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Pair-level 门控融合
        
        Args:
            ho_tokens: [B, N_pairs, D] 人-物对特征
            cls_token: [B, D] 全局特征 (class_embedding)
        
        Returns:
            fused_tokens: [B, N_pairs, D] 融合后的特征
        """
        B, N, D = ho_tokens.shape
        
        cls_expanded = cls_token.unsqueeze(1).expand(-1, N, -1)
        
        combined = torch.cat([ho_tokens, cls_expanded], dim=-1)
        
        gate = self.pair_gate_mlp(combined)
        
        fused_tokens = ho_tokens + gate * cls_expanded
        
        return fused_tokens


def build_scene_gate(gate_type: str = 'image', dim: int = 512, hidden_dim: int = 128):
    if gate_type == 'image':
        return SceneGate(dim=dim, hidden_dim=hidden_dim)
    elif gate_type == 'pair':
        return SceneGatePair(dim=dim, hidden_dim=hidden_dim * 2)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}. Supported: 'image', 'pair'")
