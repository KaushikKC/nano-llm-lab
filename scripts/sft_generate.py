"""Generate before/after responses for eval.jsonl prompts.

Runs each prompt through (a) the raw base model and (b) the SFT checkpoint,
then writes a side-by-side markdown table to docs/sft/before_after.md.

Usage:
    python scripts/sft_generate.py \
        --base  Qwen/Qwen2.5-0.5B \
        --sft   checkpoints/sft/hf \
        --eval  data/sft/eval.jsonl \
        --out   docs/sft/before_after.md
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanolab.sft.chat_format import build_full_text


# ─── helpers ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def generate_response(
    row: dict,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 300,
) -> str:
    prompt_str, _ = build_full_text(row, tokenizer)
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)

    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # greedy — deterministic
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Decode only the newly generated tokens
    new_ids = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _md_cell(text: str, width: int = 80) -> str:
    """Wrap text for a markdown table cell, escaping pipe characters."""
    text = text.replace("|", "\\|").replace("\n", " ↵ ")
    return textwrap.shorten(text, width=width, placeholder="…")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT before/after generation")
    parser.add_argument("--base",  default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft",   default="checkpoints/sft/hf")
    parser.add_argument("--eval",  default="data/sft/eval.jsonl")
    parser.add_argument("--out",   default="docs/sft/before_after.md")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    rows = [json.loads(l) for l in Path(args.eval).read_text().splitlines() if l.strip()]

    print(f"Loading base model: {args.base}")
    base_tok, base_model = load_model(args.base, device)

    print(f"Loading SFT model: {args.sft}")
    sft_tok, sft_model = load_model(args.sft, device)

    lines: list[str] = []
    lines.append("# SFT before/after generations\n")
    lines.append(
        "Comparison of the raw Qwen2.5-0.5B base model versus the same model after "
        "supervised fine-tuning on the Solidity/DeFi dataset. "
        "Decoding: greedy (`do_sample=False`), `max_new_tokens=300`.\n"
    )

    for i, row in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] {row['category']}: {row['user'][:60]}…")

        base_out = generate_response(row, base_tok, base_model, device, args.max_new_tokens)
        sft_out  = generate_response(row, sft_tok,  sft_model,  device, args.max_new_tokens)

        lines.append(f"---\n\n## Example {i+1} — `{row['category']}`\n")
        lines.append(f"**Prompt**: {row['user'][:200]}{'…' if len(row['user']) > 200 else ''}\n")
        lines.append("\n| | Response |\n|---|---|\n")
        lines.append(f"| **Base (before SFT)** | {_md_cell(base_out)} |\n")
        lines.append(f"| **SFT (after)**       | {_md_cell(sft_out)} |\n")
        lines.append("\n<details><summary>Full responses</summary>\n\n")
        lines.append(f"**Base:**\n```\n{base_out}\n```\n\n")
        lines.append(f"**SFT:**\n```\n{sft_out}\n```\n\n")
        lines.append("</details>\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
