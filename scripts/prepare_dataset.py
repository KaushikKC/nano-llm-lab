"""Tokenize the TinyStories text files and pack them into flat token-id
binaries for fast memmap-based random access during training.

Every story is tokenized and followed by an `<|endoftext|>` token, then all
token ids are concatenated into one long 1-D array and written as
`np.uint16` (vocab_size=8192 fits comfortably in 16 bits). Lines are
processed in fixed-size chunks and written incrementally so memory usage
stays bounded regardless of dataset size (the 2.1M-line TinyStories train
split otherwise blows up to tens of GB if `encode_batch` is called on the
whole file at once). A small `meta.json` records the vocab size and token
counts.

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
CHUNK_LINES = 10_000


def _encode_chunk(tokenizer: Tokenizer, lines: list[str], eos_id: int) -> np.ndarray:
    encoded = tokenizer.encode_batch(lines, add_special_tokens=False)
    chunks = []
    for enc in encoded:
        chunks.append(np.array(enc.ids, dtype=np.uint16))
        chunks.append(np.array([eos_id], dtype=np.uint16))
    return np.concatenate(chunks)


def tokenize_file(tokenizer: Tokenizer, in_path: str, out_path: str) -> int:
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    with open(in_path, encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)

    total_tokens = 0
    with open(in_path, encoding="utf-8") as f_in, open(out_path, "wb") as f_out:
        chunk: list[str] = []
        with tqdm(total=num_lines, desc=f"Tokenizing {os.path.basename(in_path)}") as pbar:
            for line in f_in:
                chunk.append(line)
                if len(chunk) >= CHUNK_LINES:
                    tokens = _encode_chunk(tokenizer, chunk, eos_id)
                    f_out.write(tokens.tobytes())
                    total_tokens += len(tokens)
                    pbar.update(len(chunk))
                    chunk = []
            if chunk:
                tokens = _encode_chunk(tokenizer, chunk, eos_id)
                f_out.write(tokens.tobytes())
                total_tokens += len(tokens)
                pbar.update(len(chunk))

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
