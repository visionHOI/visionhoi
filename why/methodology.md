# Methodology

## 3. Methodology

### 3.1 Scene-Aware Gate Mechanism

#### 3.1.1 Motivation

While LA and IA capture locality and interaction information, the original CLIP architecture discards the global scene context. In the standard CLIP, the global feature `x[:,:-196,:]` contains both HO tokens and the class embedding (cls_token). The class embedding represents the overall scene of the entire image. However, in our modified architecture (clip_v1.py), we split the return values into three separate outputs: `ho_tokens`, `cls_token`, and `local_feat`. While `cls_token` is preserved, it is not directly utilized to represent the global scene context. For example, consider an image of "a person holding a book". The HO token for this pair only knows there is a "person" and a "book" in the image, but has no idea whether this is happening in a library (reading), a bookstore (buying), or a classroom (studying). Without global scene information, the model cannot distinguish between these scenarios, leading to poor generalization on unseen HOI categories. To address this, we propose a lightweight scene-aware gating mechanism that dynamically incorporates global scene information (via cls_token) into instance-level representations.

#### 3.1.2 Architecture

SceneGate is a lightweight gating module designed to fuse global scene features with local instance-level features. The core idea is simple yet effective:

1. **Input**: Takes the image-level [CLS] token (representing the global scene) and instance-level HO tokens (representing individual human-object pairs).

2. **Gating Mechanism**: Uses a small MLP (multi-layer perceptron) to learn how much each HO token should be influenced by the global scene context. The MLP outputs a scalar value between -1 and 1 (via Tanh activation), representing the gating weight.

3. **Feature Fusion**: Applies the learned gate weight to modulate the contribution of the global scene feature (cls_token) to each HO token. This allows the model to dynamically adjust the influence of scene context based on the specific instance.

The key advantage is that SceneGate is extremely lightweight (~65K parameters) and can be easily integrated into existing architectures without significant computational overhead.

#### 3.1.3 Forward Pass

Given the instance-level HO tokens ho_tokens ∈ [B, N_pairs, D] and the global scene feature cls_token ∈ [B, D], SceneGate computes the gating weights and fuses the features as follows:

gate = σ(MLP(cls_token))  # [B, 1]
fused_tokens = ho_tokens + gate ⊙ cls_token  # [B, N_pairs, D]

where σ denotes the Tanh activation function and ⊙ denotes element-wise multiplication.

#### 3.1.4 Zero-Initialization

To ensure training stability, we apply zero-initialization to the last layer of the gate MLP:

```python
last_layer = self.gate_mlp[-2]
nn.init.zeros_(last_layer.weight)
nn.init.zeros_(last_layer.bias)
```

This ensures that during the initial training phase, the gating mechanism has no effect, and the model behaves similarly to the baseline. The scene-aware capability is then gradually learned as training progresses.

#### 3.1.5 Intuition

The global cls_token provides scene context (indoor/outdoor, scene type, etc.), while the gating weights dynamically adjust the importance of each instance feature. The fusion operation ho_tokens + gate * cls_token preserves the original features while injecting scene information, enhancing the model's generalization capability on unseen HOI categories.

---

### 3.2 Low-Rank Decomposition Analysis

#### 3.2.1 Motivation

To further improve parameter efficiency and prevent overfitting, we explore Low-Rank decomposition for the SceneGate module. Low-Rank decomposition reduces the number of parameters from O(D²) to O(2Dr), where r << D is the rank of the decomposition.

#### 3.2.2 Architecture

The gate weight matrix W is decomposed into two smaller matrices:

W = W_down · W_up  # W_down ∈ [D, r], W_up ∈ [r, D]

where r is the rank (set to 16 in our experiments). The forward pass becomes:

gate = σ(W_up · ReLU(W_down · cls_token))

#### 3.2.3 Experimental Findings

Despite the significant parameter reduction (95% fewer parameters), Low-Rank decomposition leads to performance degradation at later epochs. The possible reasons include:

1. **Limited Expressive Capacity**: Low-Rank decomposition restricts the model's ability to learn complex scene-interaction relationships.
2. **Excessive Compression**: Scene information is over-compressed, losing important details.
3. **Poor Generalization**: The reduced capacity hinders the model's ability to generalize to unseen HOI categories.

**Conclusion**: While Low-Rank decomposition offers parameter efficiency, it significantly harms performance. The full-rank SceneGate achieves better results and is recommended for production use.