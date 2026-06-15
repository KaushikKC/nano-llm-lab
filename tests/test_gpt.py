import math

import pytest
import torch

from nanolab.model.gpt import GPT


def _expected_param_count(cfg, non_embedding: bool = False) -> int:
    embed = cfg.vocab_size * cfg.d_model  # tied with lm_head, counted once
    per_block = (
        4 * cfg.d_model**2  # attention: qkv_proj (3x) + out_proj
        + 3 * cfg.d_model * cfg.d_ff  # SwiGLU: w_gate + w_up + w_down
        + 2 * cfg.d_model  # two RMSNorm weight vectors
    )
    final_norm = cfg.d_model
    total = embed + cfg.n_layer * per_block + final_norm
    return total - embed if non_embedding else total


def test_logits_shape_and_no_loss_without_targets(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (2, 10))
    logits, loss = model(idx)
    assert logits.shape == (2, 10, tiny_config.vocab_size)
    assert loss is None


def test_loss_with_targets_is_finite_and_near_uniform_at_init(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (2, 10))
    targets = torch.randint(0, tiny_config.vocab_size, (2, 10))
    _, loss = model(idx, targets)

    assert loss is not None
    assert torch.isfinite(loss)
    # At initialization the model's output distribution is close to uniform,
    # so cross-entropy should be close to ln(vocab_size).
    assert abs(loss.item() - math.log(tiny_config.vocab_size)) < 1.0


def test_weight_tying(tiny_config):
    model = GPT(tiny_config)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_param_count_matches_formula(tiny_config):
    model = GPT(tiny_config)
    assert model.num_params() == _expected_param_count(tiny_config)
    assert model.num_params(non_embedding=True) == _expected_param_count(
        tiny_config, non_embedding=True
    )


def test_sequence_longer_than_max_seq_len_raises(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.max_seq_len + 1))
    with pytest.raises(ValueError):
        model(idx)


def test_overfits_a_single_tiny_batch(tiny_config):
    """The cheapest possible end-to-end check: forward, backward, and the
    optimizer step are all wired correctly if a tiny model can memorize a
    single fixed batch within a hundred steps."""
    model = GPT(tiny_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    idx = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.max_seq_len))
    targets = torch.randint(0, tiny_config.vocab_size, (4, tiny_config.max_seq_len))

    _, initial_loss = model(idx, targets)

    for _ in range(100):
        _, loss = model(idx, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, final_loss = model(idx, targets)
    assert final_loss.item() < 0.5 * initial_loss.item()
