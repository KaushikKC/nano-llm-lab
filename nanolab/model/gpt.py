import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanolab.config import GPTConfig
from nanolab.model.block import TransformerBlock
from nanolab.model.norm import RMSNorm


class GPT(nn.Module):
    """A decoder-only transformer language model.

    Architecture: token embedding -> N x TransformerBlock (RoPE attention +
    SwiGLU, pre-norm RMSNorm, residuals) -> final RMSNorm -> linear LM head.
    There is no separate learned positional embedding table — position
    information enters entirely through RoPE inside each attention layer.
    The LM head shares its weight matrix with the token embedding ("weight
    tying"), which roughly halves the parameter cost of the vocabulary and
    acts as a regularizer (the same matrix must work for both "look up a
    token's vector" and "score every token against a hidden state").
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_head=config.n_head,
                    d_ff=config.d_ff,
                    max_seq_len=config.max_seq_len,
                    rope_base=config.rope_base,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: the LM head and the token embedding share parameters.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Extra scaling for the projections that write directly into the
        # residual stream, so their variance doesn't grow with depth
        # (GPT-2 / nanoGPT convention).
        residual_scale = 1.0 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * residual_scale)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=0.02 * residual_scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_emb.weight.numel()
        return n_params
