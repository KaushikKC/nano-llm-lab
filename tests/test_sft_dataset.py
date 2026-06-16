"""Tests for nanolab.sft.dataset.SFTDataset and collate_fn."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
import pytest

from nanolab.sft.dataset import SFTDataset, collate_fn


# ─── helpers ──────────────────────────────────────────────────────────────────

SAMPLE_ROW = {
    "category": "vulnerability_id",
    "system": "You are a smart contract auditor.",
    "user": "Is this safe?\n\n```solidity\nfunction foo() external { bar(); }\n```",
    "assistant": "No, missing access control. Add onlyOwner.",
}


def _make_tokenizer(vocab_size: int = 256, max_len: int = 512) -> MagicMock:
    """Simple deterministic mock tokenizer: encodes each char as its ord % vocab_size."""
    tok = MagicMock()
    tok.chat_template = None  # use fallback so no complex mock needed
    tok.eos_token = "<EOS>"
    tok.eos_token_id = 1
    tok.pad_token_id = 0

    class FakeEncoding:
        def __init__(self, ids):
            self.input_ids = torch.tensor([ids])  # shape [1, seq]

    def call(text, truncation=True, max_length=512, return_tensors="pt"):
        ids = [ord(c) % vocab_size for c in text]
        if truncation:
            ids = ids[:max_length]
        return FakeEncoding(ids)

    tok.side_effect = call
    return tok


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ─── SFTDataset ───────────────────────────────────────────────────────────────

def test_len():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.jsonl"
        _write_jsonl([SAMPLE_ROW] * 5, p)
        ds = SFTDataset(p, _make_tokenizer())
    assert len(ds) == 5


def test_item_has_correct_keys():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.jsonl"
        _write_jsonl([SAMPLE_ROW], p)
        ds = SFTDataset(p, _make_tokenizer())
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item


def test_input_and_labels_same_length():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.jsonl"
        _write_jsonl([SAMPLE_ROW], p)
        ds = SFTDataset(p, _make_tokenizer())
    item = ds[0]
    assert item["input_ids"].shape == item["labels"].shape


def test_prompt_tokens_are_masked():
    """labels[:prompt_len] must all be -100."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.jsonl"
        _write_jsonl([SAMPLE_ROW], p)
        ds = SFTDataset(p, _make_tokenizer())
    labels = ds[0]["labels"]
    # There must be at least one -100 at the start (prompt)
    assert labels[0].item() == -100, "First label token (prompt) must be masked"


def test_some_labels_not_masked():
    """At least one assistant token should have a real label."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.jsonl"
        _write_jsonl([SAMPLE_ROW], p)
        ds = SFTDataset(p, _make_tokenizer())
    labels = ds[0]["labels"]
    assert (labels != -100).any(), "At least one label must be unmasked (assistant tokens)"


def test_input_ids_are_long_tensors():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.jsonl"
        _write_jsonl([SAMPLE_ROW], p)
        ds = SFTDataset(p, _make_tokenizer())
    assert ds[0]["input_ids"].dtype == torch.long
    assert ds[0]["labels"].dtype == torch.long


def test_truncation_to_max_seq_len():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.jsonl"
        _write_jsonl([SAMPLE_ROW], p)
        ds = SFTDataset(p, _make_tokenizer(), max_seq_len=32)
    assert ds[0]["input_ids"].shape[0] <= 32


# ─── collate_fn ───────────────────────────────────────────────────────────────

def test_collate_pads_to_max_length():
    items = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 4, 5])},
        {"input_ids": torch.tensor([6, 7]),    "labels": torch.tensor([-100, 8])},
    ]
    batch = collate_fn(items, pad_id=0)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)


def test_collate_pads_input_ids_with_pad_id():
    items = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 4, 5])},
        {"input_ids": torch.tensor([6]),        "labels": torch.tensor([-100])},
    ]
    batch = collate_fn(items, pad_id=99)
    # Shorter item (row 1) should be padded with 99
    assert batch["input_ids"][1, 1].item() == 99
    assert batch["input_ids"][1, 2].item() == 99


def test_collate_pads_labels_with_minus_100():
    items = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 4, 5])},
        {"input_ids": torch.tensor([6]),        "labels": torch.tensor([-100])},
    ]
    batch = collate_fn(items, pad_id=0)
    assert batch["labels"][1, 1].item() == -100
    assert batch["labels"][1, 2].item() == -100


def test_collate_single_item():
    items = [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, 3])}]
    batch = collate_fn(items, pad_id=0)
    assert batch["input_ids"].shape == (1, 2)
    assert batch["labels"].shape == (1, 2)
