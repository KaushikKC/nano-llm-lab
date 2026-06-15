"""Download the TinyStories dataset and dump it to plain text files.

TinyStories (Eldan & Li, 2023, https://huggingface.co/datasets/roneneldan/TinyStories)
is a synthetic dataset of short stories written by GPT-3.5/GPT-4, using a
restricted ~1,500-word vocabulary intended to be learnable by very small
language models. The "train" split has ~2.1M stories, "validation" has ~22K.

Output:
    data/raw/train.txt — one story per line
    data/raw/valid.txt — one story per line

Usage:
    python scripts/download_data.py [--out-dir data/raw]
"""

import argparse
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split, filename in [("train", "train.txt"), ("validation", "valid.txt")]:
        out_path = os.path.join(args.out_dir, filename)
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping download for split '{split}'")
            continue

        print(f"Downloading TinyStories split '{split}'...")
        ds = load_dataset("roneneldan/TinyStories", split=split)

        print(f"Writing {len(ds)} stories to {out_path}...")
        with open(out_path, "w", encoding="utf-8") as f:
            for example in ds:
                # Collapse each story to a single line so the downstream
                # tokenizer/packing scripts can treat one line == one document.
                text = example["text"].replace("\n", " ").strip()
                if text:
                    f.write(text + "\n")

        print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
