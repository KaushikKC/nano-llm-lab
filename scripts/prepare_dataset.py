"""Tokenize the TinyStories text files and pack them into flat token-id
binaries for fast memmap-based random access during training.

For each split, every story is tokenized and followed by an
`<|endoftext|>` token, then all token ids are concatenated into one long
1-D array and written as `np.uint16` (vocab_size=8192 fits comfortably in
16 bits). A small `meta.json` records the vocab size and token counts.

Output:
    data/processed/train.bin
    data/processed/val.bin
    data/processed/meta.json

Usage:
    python scripts/prepare_dataset.py
"""

import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

EOS_TOKEN = "<|endoftext|>"


def tokenize_file(tokenizer: Tokenizer, in_path: str, out_path: str) -> int:
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    # First pass: count tokens so we can pre-allocate the memmap.
    total_tokens = 0
    with open(in_path, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Tokenizing {len(lines)} lines from {in_path}...")
    encoded_lines = tokenizer.encode_batch(lines, add_special_tokens=False)
    for enc in encoded_lines:
        total_tokens += len(enc.ids) + 1  # +1 for EOS

    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    idx = 0
    for enc in tqdm(encoded_lines, desc=f"Packing {os.path.basename(out_path)}"):
        ids = enc.ids
        arr[idx : idx + len(ids)] = ids
        idx += len(ids)
        arr[idx] = eos_id
        idx += 1
    arr.flush()
    assert idx == total_tokens
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    counts = {}
    for split, in_name, out_name in [
        ("train", "train.txt", "train.bin"),
        ("val", "valid.txt", "val.bin"),
    ]:
        in_path = os.path.join(args.raw_dir, in_name)
        out_path = os.path.join(args.out_dir, out_name)
        counts[split] = tokenize_file(tokenizer, in_path, out_path)
        print(f"{split}: {counts[split]:,} tokens -> {out_path}")

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_token_id": tokenizer.token_to_id(EOS_TOKEN),
        "token_counts": counts,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}: {meta}")


if __name__ == "__main__":
    main()
