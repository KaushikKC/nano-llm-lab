import torch


def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute the cos/sin tables used by rotary positional embeddings.

    RoPE encodes absolute position by rotating each consecutive pair of
    dimensions in the query/key vectors by an angle that grows linearly with
    position, at a frequency that varies geometrically across pairs (low
    pairs rotate fast, high pairs rotate slowly — like a clock with many
    hands). Because rotation is linear, the dot product of two rotated
    vectors depends only on the *difference* in their positions, which is
    what gives attention its relative-position sensitivity "for free".

    Returns:
        cos, sin: each of shape (max_seq_len, head_dim // 2)
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    # One frequency per pair of dimensions: theta_i = base ** (-2i / head_dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    # Outer product: angle(pos, i) = pos * theta_i
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to query/key tensors.

    Args:
        x: (..., seq_len, head_dim)
        cos, sin: (seq_len, head_dim // 2), sliced from precompute_rope_freqs

    Returns:
        Tensor of the same shape as x, with each consecutive pair of
        dimensions rotated by the corresponding position-dependent angle
        ("rotate-half" convention: pairs are (x[..., :d/2], x[..., d/2:])).
    """
    head_dim = x.shape[-1]
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]

    # Broadcast cos/sin over any leading dims (batch, heads, ...)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x2 * cos + x1 * sin
    return torch.cat([rotated_x1, rotated_x2], dim=-1)
