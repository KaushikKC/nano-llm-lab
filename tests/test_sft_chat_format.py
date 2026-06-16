"""Tests for nanolab.sft.chat_format.build_full_text."""

import pytest
from unittest.mock import MagicMock

from nanolab.sft.chat_format import build_full_text


# ─── helpers ──────────────────────────────────────────────────────────────────

SAMPLE_ROW = {
    "category": "vulnerability_id",
    "system": "You are a smart contract auditor.",
    "user": "Is this code safe?",
    "assistant": "No, it has a reentrancy bug.",
}


def _make_tokenizer_with_template() -> MagicMock:
    """Minimal mock that behaves like Qwen2.5's tokenizer (has chat_template)."""
    tok = MagicMock()
    tok.chat_template = "{% for msg in messages %}..."  # truthy
    tok.eos_token = "<|endoftext|>"

    def apply_chat_template(messages, tokenize, add_generation_prompt):
        sys_msg = next(m["content"] for m in messages if m["role"] == "system")
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        base = (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        )
        if add_generation_prompt:
            return base + "<|im_start|>assistant\n"
        asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        return base + f"<|im_start|>assistant\n{asst_msg}<|im_end|>\n"

    tok.apply_chat_template.side_effect = apply_chat_template
    return tok


def _make_tokenizer_no_template() -> MagicMock:
    """Tokenizer without a chat_template (triggers the fallback path)."""
    tok = MagicMock()
    tok.chat_template = None
    tok.eos_token = "<|endoftext|>"
    return tok


# ─── tests ────────────────────────────────────────────────────────────────────

def test_prompt_is_prefix_of_full_with_template():
    tok = _make_tokenizer_with_template()
    prompt, full = build_full_text(SAMPLE_ROW, tok)
    assert full.startswith(prompt), "full_str must begin with prompt_str"


def test_full_contains_assistant_answer_with_template():
    tok = _make_tokenizer_with_template()
    _, full = build_full_text(SAMPLE_ROW, tok)
    assert SAMPLE_ROW["assistant"] in full


def test_full_ends_with_eos_with_template():
    tok = _make_tokenizer_with_template()
    _, full = build_full_text(SAMPLE_ROW, tok)
    assert full.endswith(tok.eos_token), "full_str must end with eos_token"


def test_prompt_does_not_contain_assistant_answer_with_template():
    tok = _make_tokenizer_with_template()
    prompt, _ = build_full_text(SAMPLE_ROW, tok)
    assert SAMPLE_ROW["assistant"] not in prompt


def test_fallback_prompt_is_prefix_of_full():
    tok = _make_tokenizer_no_template()
    prompt, full = build_full_text(SAMPLE_ROW, tok)
    assert full.startswith(prompt)


def test_fallback_full_contains_assistant_answer():
    tok = _make_tokenizer_no_template()
    _, full = build_full_text(SAMPLE_ROW, tok)
    assert SAMPLE_ROW["assistant"] in full


def test_fallback_full_ends_with_eos():
    tok = _make_tokenizer_no_template()
    _, full = build_full_text(SAMPLE_ROW, tok)
    assert full.endswith(tok.eos_token)


def test_system_and_user_in_prompt():
    tok = _make_tokenizer_with_template()
    prompt, _ = build_full_text(SAMPLE_ROW, tok)
    assert SAMPLE_ROW["system"] in prompt
    assert SAMPLE_ROW["user"] in prompt


def test_fallback_system_and_user_in_prompt():
    tok = _make_tokenizer_no_template()
    prompt, _ = build_full_text(SAMPLE_ROW, tok)
    assert SAMPLE_ROW["system"] in prompt
    assert SAMPLE_ROW["user"] in prompt
