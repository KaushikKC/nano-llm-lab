import torch

from nanolab.model.norm import RMSNorm


def test_output_shape_matches_input():
    norm = RMSNorm(dim=16)
    x = torch.randn(4, 8, 16)
    assert norm(x).shape == x.shape


def test_rms_is_normalized_to_unit_with_default_weight():
    norm = RMSNorm(dim=32)
    x = torch.randn(8, 32) * 5.0  # arbitrary scale
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_scale_invariance():
    norm = RMSNorm(dim=32)
    x = torch.randn(8, 32)
    out1 = norm(x)
    out2 = norm(x * 10.0)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_gradients_flow():
    norm = RMSNorm(dim=16)
    x = torch.randn(4, 16, requires_grad=True)
    out = norm(x)
    out.sum().backward()
    assert x.grad is not None
    assert norm.weight.grad is not None
