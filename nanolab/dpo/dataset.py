"""DPODataset: tokenises (prompt, chosen, rejected) preference triples.

Each JSONL row must have: system, prompt, chosen, rejected.
The dataset produces one example per row as a dict with keys:
  input_ids_w, labels_w      -- chosen sequence, labels masked on prompt
  input_ids_l, labels_l      -- rejected sequence, labels masked on prompt
  prompt_len                 -- number of prompt tokens (same for both)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DPODataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int) -> None:
        self.examples = []
        raw = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

        for row in raw:
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system",  "content": row["system"]},
                    {"role": "user",    "content": row["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            chosen_text  = prompt_text + row["chosen"]  + tokenizer.eos_token
            rejected_text = prompt_text + row["rejected"] + tokenizer.eos_token

            prompt_ids   = tokenizer(prompt_text,   add_special_tokens=False)["input_ids"]
            chosen_ids   = tokenizer(chosen_text,   add_special_tokens=False)["input_ids"]
            rejected_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"]

            # Truncate if needed
            chosen_ids   = chosen_ids[:max_seq_len]
            rejected_ids = rejected_ids[:max_seq_len]
            prompt_len   = min(len(prompt_ids), max_seq_len)

            def make_labels(ids, plen):
                lbl = list(ids)
                for i in range(plen):
                    lbl[i] = -100
                return lbl

            self.examples.append({
                "input_ids_w": chosen_ids,
                "labels_w":    make_labels(chosen_ids, prompt_len),
                "input_ids_l": rejected_ids,
                "labels_l":    make_labels(rejected_ids, prompt_len),
                "prompt_len":  prompt_len,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def dpo_collate(batch: list[dict], pad_id: int) -> dict:
    """Collate a list of DPO examples, padding to max length within the batch."""
    def _pad(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    return {
        "input_ids_w": _pad([b["input_ids_w"] for b in batch], pad_id),
        "labels_w":    _pad([b["labels_w"]    for b in batch], -100),
        "input_ids_l": _pad([b["input_ids_l"] for b in batch], pad_id),
        "labels_l":    _pad([b["labels_l"]    for b in batch], -100),
    }
