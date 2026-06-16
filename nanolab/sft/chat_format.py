"""Utilities to render dataset rows into prompt strings using the tokenizer's
chat_template (ChatML on Qwen2.5) and to return the byte-length of the prompt
portion so labels can be masked before the assistant's tokens."""

from __future__ import annotations

from typing import Any


_CHATML_FALLBACK = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_full_text(row: dict[str, Any], tokenizer: Any) -> tuple[str, str]:
    """Return (prompt_str, full_str) for a dataset row.

    prompt_str : everything up to (and including) the assistant turn opener
    full_str   : prompt_str + assistant answer + EOS

    Uses tokenizer.apply_chat_template when available (Qwen2.5 ships ChatML
    on the base checkpoint); falls back to an inline ChatML template otherwise.
    """
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user",   "content": row["user"]},
    ]
    full_messages = messages + [{"role": "assistant", "content": row["assistant"]}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_str = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # apply_chat_template doesn't append EOS for the completed turn on all
        # versions; ensure it ends with the tokenizer's eos_token.
        eos = tokenizer.eos_token or "<|endoftext|>"
        if not full_str.endswith(eos):
            full_str = full_str + eos
    else:
        prompt_str = _CHATML_FALLBACK.format(
            system=row["system"], user=row["user"]
        )
        eos = getattr(tokenizer, "eos_token", "<|endoftext|>")
        full_str = prompt_str + row["assistant"] + "<|im_end|>\n" + eos

    return prompt_str, full_str
