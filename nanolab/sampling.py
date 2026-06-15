import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Autoregressively extend `idx` by `max_new_tokens` tokens.

    Args:
        model: a GPT model returning (logits, loss) from forward(idx).
        idx: (B, T) starting token ids.
        max_new_tokens: number of tokens to generate.
        temperature: softmax temperature. 0.0 means greedy (argmax) decoding.
        top_k: if set, restrict sampling to the top-k highest-probability tokens.
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
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return idx
