import torch

from nanolab.model.mlp import SwiGLU


def test_output_shape_matches_input():
    mlp = SwiGLU(d_model=16, d_ff=48)
    x = torch.randn(4, 8, 16)
    assert mlp(x).shape == x.shape


def test_zero_input_gives_zero_output():
    """silu(0) == 0, so silu(w_gate(0)) * w_up(0) == 0 * 0 == 0 regardless of
    the (no-bias) weight values — this would NOT hold if the gate and value
    streams were combined by addition instead of multiplication."""
    mlp = SwiGLU(d_model=16, d_ff=48)
    x = torch.zeros(2, 16)
    out = mlp(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_param_count_matches_formula():
    d_model, d_ff = 16, 48
    mlp = SwiGLU(d_model, d_ff)
    expected = 3 * d_model * d_ff  # w_gate + w_up + w_down, no biases
    actual = sum(p.numel() for p in mlp.parameters())
    assert actual == expected


def test_gradients_flow_to_all_three_projections():
    mlp = SwiGLU(d_model=16, d_ff=48)
    x = torch.randn(4, 16, requires_grad=True)
    mlp(x).sum().backward()
    assert mlp.w_gate.weight.grad is not None
    assert mlp.w_up.weight.grad is not None
    assert mlp.w_down.weight.grad is not None
