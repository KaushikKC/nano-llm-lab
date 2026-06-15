import pytest
import torch

from nanolab.model.attention import CausalSelfAttention


def test_output_shape_matches_input():
    attn = CausalSelfAttention(d_model=32, n_head=4, max_seq_len=16)
    x = torch.randn(2, 10, 32)
    assert attn(x).shape == x.shape


def test_invalid_head_split_raises():
    with pytest.raises(ValueError):
        CausalSelfAttention(d_model=33, n_head=4, max_seq_len=16)


def test_causality_future_tokens_do_not_affect_past_outputs():
    """Changing tokens at positions >= t must not change the output at
    position t — this is the single most important correctness property of
    a causal attention layer."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=16, n_head=2, max_seq_len=8)
    attn.eval()

    x = torch.randn(1, 8, 16)
    x_modified = x.clone()
    x_modified[:, 4:, :] = torch.randn(1, 4, 16)  # change everything from position 4 onward

    out = attn(x)
    out_modified = attn(x_modified)

    # Positions 0..3 are unaffected by the change at positions 4..7.
    assert torch.allclose(out[:, :4, :], out_modified[:, :4, :], atol=1e-6)
    # Position 4 onward should generally differ.
    assert not torch.allclose(out[:, 4:, :], out_modified[:, 4:, :])


def test_gradients_flow_to_projections():
    attn = CausalSelfAttention(d_model=32, n_head=4, max_seq_len=16)
    x = torch.randn(2, 10, 32, requires_grad=True)
    attn(x).sum().backward()
    assert attn.qkv_proj.weight.grad is not None
    assert attn.out_proj.weight.grad is not None
