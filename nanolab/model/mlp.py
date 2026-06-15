import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated MLP using the SwiGLU activation (Shazeer, 2020).

    Instead of a single up-projection followed by a pointwise activation
    (e.g. GELU), SwiGLU computes two parallel projections of the input — a
    "gate" and a "value" — and multiplies SiLU(gate) elementwise with value
    before the down-projection. The gate lets the network learn, per
    activation, how much of the value stream to let through.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
