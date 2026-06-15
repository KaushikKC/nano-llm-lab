"""Generate sample completions from a trained checkpoint.

Loads a checkpoint + the project's BPE tokenizer, then generates completions
for a handful of prompts across several sampling temperatures. Useful both
as a sanity check during development and to produce the README sample
outputs.

Usage:
    python scripts/generate_samples.py --ckpt checkpoints/small/ckpt_last.pt
    python scripts/generate_samples.py --ckpt checkpoints/tiny/ckpt_last.pt \
        --max-new-tokens 100 --temperatures 0.6 0.8 1.0
"""

import argparse

import torch
from tokenizers import Tokenizer

from nanolab.model.gpt import GPT
from nanolab.sampling import generate
from nanolab.utils import get_device

DEFAULT_PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom and his dog went to the park.",
    "One day, a boy found a",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2]
    )
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    device = get_device("auto")
    print(f"Using device: {device}")

    # weights_only=False: these are checkpoints we wrote ourselves (contain a
    # GPTConfig dataclass, which torch's default weights-only unpickler rejects).
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GPT(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt['step']}")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    for temperature in args.temperatures:
        print(f"\n{'=' * 60}\nTemperature {temperature}\n{'=' * 60}")
        for prompt in args.prompts:
            ids = tokenizer.encode(prompt).ids
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = generate(
                model,
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_k=args.top_k,
                eos_token_id=eos_id,
            )
            text = tokenizer.decode(out[0].tolist())
            print(f"\nPrompt: {prompt!r}\n-> {text}")


if __name__ == "__main__":
    main()
