import torch
import torch.nn as nn
import torch.nn.functional as F

from nanolab.model.rope import apply_rope, precompute_rope_freqs


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary positional embeddings.

    The projections, head reshaping, RoPE application, and causal masking are
    all explicit so the data flow is visible end to end. The scaled dot
    product itself (matmul -> mask -> softmax -> matmul) is delegated to
    `F.scaled_dot_product_attention(..., is_causal=True)`, which is a fused
    kernel (the "flash attention" algorithm) rather than a layer abstraction
    — mathematically identical to the naive implementation but much faster.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout

        # Fused projection: produces Q, K, V in a single matmul.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)

        # (B, n_head, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
