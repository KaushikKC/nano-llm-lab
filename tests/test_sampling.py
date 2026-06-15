import torch

from nanolab.model.gpt import GPT
from nanolab.sampling import _top_p_filter, generate


def test_generate_extends_sequence_by_max_new_tokens(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (2, 4))
    out = generate(model, idx, max_new_tokens=5, temperature=1.0)
    assert out.shape == (2, 4 + 5)


def test_generate_stops_early_on_eos(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    # Greedy decoding is deterministic, so whatever token it picks first we
    # can use as the eos id to force an immediate stop.
    logits, _ = model(idx)
    eos_id = logits[0, -1, :].argmax().item()
    out = generate(model, idx, max_new_tokens=10, temperature=0.0, eos_token_id=eos_id)
    assert out.shape[1] == 5  # original 4 tokens + exactly 1 generated


def test_greedy_decoding_is_deterministic(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    out1 = generate(model, idx.clone(), max_new_tokens=5, temperature=0.0)
    out2 = generate(model, idx.clone(), max_new_tokens=5, temperature=0.0)
    assert torch.equal(out1, out2)


def test_top_k_1_matches_greedy(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    greedy = generate(model, idx.clone(), max_new_tokens=5, temperature=0.0)
    top1 = generate(model, idx.clone(), max_new_tokens=5, temperature=1.0, top_k=1)
    assert torch.equal(greedy, top1)


def test_top_k_restricts_sampling_to_k_highest_logits(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    logits, _ = model(idx)
    k = 5
    top_k_indices = set(torch.topk(logits[:, -1, :], k, dim=-1).indices[0].tolist())

    sampled = set()
    for _ in range(50):
        out = generate(model, idx.clone(), max_new_tokens=1, temperature=1.0, top_k=k)
        sampled.add(out[0, -1].item())
    assert sampled.issubset(top_k_indices)


def test_top_p_filter_keeps_minimal_nucleus():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    filtered = _top_p_filter(logits, top_p=0.8)
    kept = torch.isfinite(filtered)
    # Cumulative mass 0.5, 0.8, 0.95, 1.0 -> first three tokens are needed to
    # reach (>=) 0.8 cumulative probability; the last is dropped.
    assert kept[0].tolist() == [True, True, True, False]


def test_generate_with_top_p_runs_and_produces_valid_shape(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (2, 4))
    out = generate(model, idx, max_new_tokens=3, temperature=1.0, top_p=0.9)
    assert out.shape == (2, 4 + 3)
