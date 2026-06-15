import pytest
import torch

from nanolab.model.rope import apply_rope, precompute_rope_freqs

HEAD_DIM = 8
MAX_SEQ = 16


def test_table_shapes():
    cos, sin = precompute_rope_freqs(HEAD_DIM, MAX_SEQ)
    assert cos.shape == (MAX_SEQ, HEAD_DIM // 2)
    assert sin.shape == (MAX_SEQ, HEAD_DIM // 2)


def test_odd_head_dim_raises():
    with pytest.raises(ValueError):
        precompute_rope_freqs(7, MAX_SEQ)


def test_output_shape_preserved():
    cos, sin = precompute_rope_freqs(HEAD_DIM, MAX_SEQ)
    x = torch.randn(2, 3, MAX_SEQ, HEAD_DIM)  # (batch, heads, seq, head_dim)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_rotation_preserves_norm():
    cos, sin = precompute_rope_freqs(HEAD_DIM, MAX_SEQ)
    x = torch.randn(5, MAX_SEQ, HEAD_DIM)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)


def test_rotation_depends_on_position():
    cos, sin = precompute_rope_freqs(HEAD_DIM, MAX_SEQ)
    x = torch.randn(1, HEAD_DIM)
    out_pos0 = apply_rope(x, cos[0:1], sin[0:1])
    out_pos5 = apply_rope(x, cos[5:6], sin[5:6])
    assert not torch.allclose(out_pos0, out_pos5)


def test_relative_position_dot_product_invariance():
    """The defining property of RoPE: <RoPE(q, i), RoPE(k, j)> depends only
    on (i - j), not on the absolute positions i and j."""
    cos, sin = precompute_rope_freqs(HEAD_DIM, MAX_SEQ)
    q = torch.randn(1, HEAD_DIM)
    k = torch.randn(1, HEAD_DIM)

    q_at_0 = apply_rope(q, cos[0:1], sin[0:1])
    k_at_3 = apply_rope(k, cos[3:4], sin[3:4])
    dot_offset_3_a = (q_at_0 * k_at_3).sum()

    q_at_5 = apply_rope(q, cos[5:6], sin[5:6])
    k_at_8 = apply_rope(k, cos[8:9], sin[8:9])
    dot_offset_3_b = (q_at_5 * k_at_8).sum()

    assert torch.allclose(dot_offset_3_a, dot_offset_3_b, atol=1e-5)
