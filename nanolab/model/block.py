import torch
import torch.nn as nn

from nanolab.model.attention import CausalSelfAttention
from nanolab.model.mlp import SwiGLU
from nanolab.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """A single pre-norm decoder block: attention sub-layer + MLP sub-layer,
    each wrapped in RMSNorm and a residual ("skip") connection.

    Pre-norm (normalize *before* the sub-layer, then add the result back to
    the un-normalized residual stream) is the modern convention — it keeps
    the residual stream's scale stable across many layers, which is what
    lets these networks be trained without learning-rate warmup tricks
    breaking down at depth.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len, rope_base, dropout)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
