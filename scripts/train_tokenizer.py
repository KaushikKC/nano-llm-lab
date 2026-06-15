"""Train a byte-level BPE tokenizer on a sample of the TinyStories training set.

Byte-level BPE (the GPT-2 style) starts from raw bytes, so any UTF-8 text is
representable with no <unk> token. We train on a sample of the corpus —
TinyStories' vocabulary is small (~1,500 words), so a few hundred thousand
stories is plenty to learn stable merges without processing all 2.1M.

Output:
    tokenizer/tokenizer.json (committed to the repo — needed to reproduce
    the vocabulary used by any checkpoint)

Usage:
    python scripts/train_tokenizer.py [--vocab-size 8192] [--sample-lines 300000]
"""

import argparse
import os

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

EOS_TOKEN = "<|endoftext|>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train.txt")
    parser.add_argument("--out", default="tokenizer/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--sample-lines", type=int, default=300_000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def sample_iterator():
        with open(args.input, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.sample_lines:
                    break
                yield line

    print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) on up to "
          f"{args.sample_lines} lines from {args.input}...")
    tokenizer.train_from_iterator(sample_iterator(), trainer=trainer)

    tokenizer.save(args.out)
    print(f"Saved tokenizer to {args.out} (vocab size = {tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
