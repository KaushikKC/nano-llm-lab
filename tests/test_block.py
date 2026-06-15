import torch

from nanolab.model.block import TransformerBlock


def test_output_shape_matches_input():
    block = TransformerBlock(d_model=32, n_head=4, d_ff=64, max_seq_len=16)
    x = torch.randn(2, 10, 32)
    assert block(x).shape == x.shape


def test_residual_identity_when_sublayers_are_zeroed():
    """If both the attention output projection and the MLP down projection
    are zeroed, x + attn(norm(x)) == x and x + mlp(norm(x)) == x, so the
    block must be an exact identity — this confirms the skip connections add
    to the input rather than replacing it."""
    block = TransformerBlock(d_model=32, n_head=4, d_ff=64, max_seq_len=16)
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.mlp.w_down.weight.zero_()

    x = torch.randn(2, 10, 32)
    out = block(x)
    assert torch.allclose(out, x)


def test_gradients_flow():
    block = TransformerBlock(d_model=32, n_head=4, d_ff=64, max_seq_len=16)
    x = torch.randn(2, 10, 32, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None
    assert block.attn.qkv_proj.weight.grad is not None
    assert block.mlp.w_gate.weight.grad is not None
