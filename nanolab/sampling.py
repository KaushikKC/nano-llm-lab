import torch
import torch.nn.functional as F


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of highest-probability tokens
    whose cumulative probability mass is >= top_p, mask out the rest."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # Tokens to remove: those *after* cumulative probability first exceeds top_p.
    sorted_remove = cumulative_probs > top_p
    # Always keep at least the single highest-probability token.
    sorted_remove[..., 0] = False
    # Shift right so the token that *crosses* top_p is kept, not removed.
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(sorted_remove).scatter(-1, sorted_idx, sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Autoregressively extend `idx` by `max_new_tokens` tokens.

    Args:
        model: a GPT model returning (logits, loss) from forward(idx).
        idx: (B, T) starting token ids.
        max_new_tokens: number of tokens to generate.
        temperature: softmax temperature. 0.0 means greedy (argmax) decoding.
        top_k: if set, restrict sampling to the top-k highest-probability tokens.
        top_p: if set, restrict sampling to the smallest nucleus of tokens whose
            cumulative probability exceeds top_p. Applied after top_k.
        eos_token_id: if every sequence in the batch produces this token, stop early.

    Returns:
        (B, T + n) tensor of token ids, where n <= max_new_tokens.
    """
    model.eval()
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, vocab_size) — logits for the next token

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth_value = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth_value, float("-inf"))
            if top_p is not None:
                logits = _top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return idx
