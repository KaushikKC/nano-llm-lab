"""Keyword-rubric evaluation script.

Generates responses from the base model and the SFT checkpoint for every example
in eval.jsonl, then scores each response by checking whether its required keywords
(the 'keywords' field in eval.jsonl) appear (case-insensitive substring match) in
the generated text. Writes a summary to docs/sft/eval_report.md.

Usage:
    python scripts/sft_eval.py \
        --base Qwen/Qwen2.5-0.5B \
        --sft  checkpoints/sft/hf \
        --eval data/sft/eval.jsonl \
        --out  docs/sft/eval_report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
def generate_response(row: dict, tokenizer, model, device: torch.device, max_new_tokens: int = 300) -> str:
    prompt_str, _ = build_full_text(row, tokenizer)
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_ids = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def keyword_score(response: str, keywords: list[str]) -> tuple[int, int]:
    """Return (hits, total) — count of keywords found in response (case-insensitive)."""
    resp_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return hits, len(keywords)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT keyword-rubric evaluation")
    parser.add_argument("--base",  default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft",   default="checkpoints/sft/hf")
    parser.add_argument("--eval",  default="data/sft/eval.jsonl")
    parser.add_argument("--out",   default="docs/sft/eval_report.md")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    rows = [json.loads(l) for l in Path(args.eval).read_text().splitlines() if l.strip()]
    assert all("keywords" in r for r in rows), "eval.jsonl rows must have 'keywords' field"

    print(f"Loading base model: {args.base}")
    base_tok, base_model = load_model(args.base, device)
    print(f"Loading SFT model: {args.sft}")
    sft_tok, sft_model = load_model(args.sft, device)

    # Per-example results
    results: list[dict] = []
    for i, row in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] {row['category']}")
        base_resp = generate_response(row, base_tok, base_model, device, args.max_new_tokens)
        sft_resp  = generate_response(row, sft_tok,  sft_model,  device, args.max_new_tokens)
        base_hits, total = keyword_score(base_resp, row["keywords"])
        sft_hits,  _     = keyword_score(sft_resp,  row["keywords"])
        results.append({
            "category": row["category"],
            "keywords": row["keywords"],
            "base_hits": base_hits,
            "sft_hits": sft_hits,
            "total": total,
            "base_resp": base_resp,
            "sft_resp": sft_resp,
        })

    # Aggregate stats
    categories = sorted(set(r["category"] for r in results))
    cat_stats: dict[str, dict] = {}
    for cat in categories:
        cat_rows = [r for r in results if r["category"] == cat]
        b_hits = sum(r["base_hits"] for r in cat_rows)
        s_hits = sum(r["sft_hits"]  for r in cat_rows)
        total  = sum(r["total"]     for r in cat_rows)
        cat_stats[cat] = {
            "n": len(cat_rows),
            "base_pct": 100 * b_hits / total if total else 0,
            "sft_pct":  100 * s_hits / total if total else 0,
        }

    overall_base = 100 * sum(r["base_hits"] for r in results) / sum(r["total"] for r in results)
    overall_sft  = 100 * sum(r["sft_hits"]  for r in results) / sum(r["total"] for r in results)

    # ── write report ──────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# SFT evaluation report — keyword coverage\n\n")
    lines.append(
        "Each eval.jsonl row has a `keywords` list. A model's response is scored "
        "by how many keywords appear in it (case-insensitive substring match). "
        "Scores are shown as `hits / total (pct%)`.\n\n"
    )

    lines.append("## Summary by category\n\n")
    lines.append("| Category | n | Base keyword% | SFT keyword% | Delta |\n")
    lines.append("|---|---|---|---|---|\n")
    for cat in categories:
        s = cat_stats[cat]
        delta = s["sft_pct"] - s["base_pct"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {cat} | {s['n']} | {s['base_pct']:.1f}% | {s['sft_pct']:.1f}% | {sign}{delta:.1f}% |\n"
        )
    lines.append(
        f"| **Overall** | {len(results)} | **{overall_base:.1f}%** | **{overall_sft:.1f}%** | "
        f"**{'+' if overall_sft >= overall_base else ''}{overall_sft - overall_base:.1f}%** |\n\n"
    )

    lines.append("## Per-example detail\n\n")
    for i, r in enumerate(results):
        b_pct = 100 * r["base_hits"] / r["total"] if r["total"] else 0
        s_pct = 100 * r["sft_hits"]  / r["total"] if r["total"] else 0
        lines.append(f"### Example {i+1} — `{r['category']}`\n\n")
        lines.append(f"**Keywords**: {', '.join(r['keywords'])}\n\n")
        lines.append(f"| | Keyword hits | Score |\n|---|---|---|\n")
        lines.append(f"| Base | {r['base_hits']}/{r['total']} | {b_pct:.0f}% |\n")
        lines.append(f"| SFT  | {r['sft_hits']}/{r['total']}  | {s_pct:.0f}% |\n\n")
        base_preview = r["base_resp"][:300].replace("\n", " ↵ ")
        sft_preview  = r["sft_resp"][:300].replace("\n", " ↵ ")
        lines.append(f"**Base response (first 300 chars)**: {base_preview}\n\n")
        lines.append(f"**SFT response (first 300 chars)**: {sft_preview}\n\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))

    print(f"\n{'='*50}")
    print(f"Base keyword coverage: {overall_base:.1f}%")
    print(f"SFT  keyword coverage: {overall_sft:.1f}%")
    print(f"Delta: {'+' if overall_sft >= overall_base else ''}{overall_sft - overall_base:.1f}%")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
