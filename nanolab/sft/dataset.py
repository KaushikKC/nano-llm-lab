"""SFTDataset: load JSONL rows, tokenize with prompt-loss masking, and collate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .chat_format import build_full_text


class SFTDataset(Dataset):
    """Tokenized SFT dataset.

    Each item is a dict with:
        input_ids : LongTensor[seq_len]   — full token sequence (prompt + assistant)
        labels    : LongTensor[seq_len]   — same as input_ids but with prompt tokens
                                            set to -100 so they don't contribute to loss
    """

    def __init__(self, path: str | Path, tokenizer: Any, max_seq_len: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        rows = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))

        self.examples: list[dict[str, torch.Tensor]] = []
        for row in rows:
            self.examples.append(self._encode(row))

    def _encode(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        prompt_str, full_str = build_full_text(row, self.tokenizer)

        full_ids = self.tokenizer(
            full_str,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).input_ids[0]

        prompt_ids = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).input_ids[0]

        prompt_len = len(prompt_ids)

        labels = full_ids.clone()
        labels[:prompt_len] = -100

        return {"input_ids": full_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int,
    fixed_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length items to the same length.

    If *fixed_len* is given, pad/truncate every item to exactly that length.
    This keeps MPS tensor shapes constant across batches, avoiding per-batch
    Metal shader re-compilation.

    input_ids are padded with pad_id; labels are padded with -100 so padding
    positions are ignored by CrossEntropyLoss.
    """
    target_len = fixed_len if fixed_len is not None else max(
        item["input_ids"].shape[0] for item in batch
    )

    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for item in batch:
        ids = item["input_ids"]
        lbs = item["labels"]
        n = ids.shape[0]
        if n < target_len:
            pad = target_len - n
            ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=torch.long)])
            lbs = torch.cat([lbs, torch.full((pad,), -100, dtype=torch.long)])
        elif n > target_len:
            ids = ids[:target_len]
            lbs = lbs[:target_len]
        input_ids_list.append(ids)
        labels_list.append(lbs)

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
    }
