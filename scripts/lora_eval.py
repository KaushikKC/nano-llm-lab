"""Keyword-rubric eval for LoRA and QLoRA adapters.

Generates responses from three models — base, LoRA, QLoRA — and scores each
against eval.jsonl keywords. Writes docs/stage3/eval_report.md.

Usage:
    python scripts/lora_eval.py \\
        --base   Qwen/Qwen2.5-0.5B \\
        --lora   checkpoints/lora/hf \\
        --qlora  checkpoints/qlora/hf \\
        --eval   data/sft/eval.jsonl \\
        --out    docs/stage3/eval_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(row: dict, tokenizer) -> str:
    messages = [
        {"role": "system",  "content": row["system"]},
        {"role": "user",    "content": row["user"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_response(model, tokenizer, prompt: str, device: torch.device,
                      max_new_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def keyword_score(response: str, keywords: list[str]) -> tuple[int, int]:
    r = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in r)
    return hits, len(keywords)


def load_model(name_or_path: str, adapter_path: Optional[str],
               device: torch.device) -> tuple:
    attn_impl = "eager" if str(device) == "mps" else "sdpa"
    base = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if adapter_path:
        base = PeftModel.from_pretrained(base, adapter_path)
        base = base.merge_and_unload()  # merge for clean inference
    base.eval()
    return base, tok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",  required=True)
    parser.add_argument("--lora",  default=None)
    parser.add_argument("--qlora", default=None)
    parser.add_argument("--eval",  required=True)
    parser.add_argument("--out",   required=True)
    args = parser.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    eval_rows = load_eval(args.eval)
    models_to_run: list[tuple[str, str | None]] = [("Base", None)]
    if args.lora:
        models_to_run.append(("LoRA", args.lora))
    if args.qlora:
        models_to_run.append(("QLoRA", args.qlora))

    # per-model results: {name: [{category, hits, total, response}]}
    results: dict[str, list[dict]] = {}

    for model_name, adapter in models_to_run:
        print(f"\nLoading {model_name} …")
        model, tokenizer = load_model(args.base, adapter, device)
        rows_out = []
        for i, row in enumerate(eval_rows):
            kws = row.get("keywords", [])
            prompt = build_prompt(row, tokenizer)
            resp = generate_response(model, tokenizer, prompt, device)
            hits, total = keyword_score(resp, kws)
            print(f"  [{i+1}/{len(eval_rows)}] {row['category']}: {hits}/{total}")
            rows_out.append({
                "category": row["category"],
                "hits": hits,
                "total": total,
                "response": resp,
            })
        results[model_name] = rows_out
        del model

    # Build report
    _write_report(args.out, eval_rows, results)
    print(f"\nReport written to: {args.out}")


def _write_report(out_path: str, eval_rows: list[dict],
                  results: dict[str, list[dict]]) -> None:
    model_names = list(results.keys())
    categories = sorted({r["category"] for r in eval_rows})

    lines = ["# Stage 3 evaluation report — keyword coverage\n"]
    lines.append("Keyword coverage per model: hits / total keywords (case-insensitive substring).\n")

    # Summary table
    lines.append("## Summary by category\n")
    header = ["Category", "n"] + model_names + [f"Δ ({model_names[-1]} − Base)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    all_hits: dict[str, int] = {m: 0 for m in model_names}
    all_total: dict[str, int] = {m: 0 for m in model_names}

    for cat in categories:
        cat_hits = {m: 0 for m in model_names}
        cat_total = {m: 0 for m in model_names}
        n = 0
        for i, row in enumerate(eval_rows):
            if row["category"] != cat:
                continue
            n += 1
            for m in model_names:
                cat_hits[m]  += results[m][i]["hits"]
                cat_total[m] += results[m][i]["total"]
        row_cells = [cat, str(n)]
        for m in model_names:
            pct = cat_hits[m] / max(1, cat_total[m]) * 100
            row_cells.append(f"{pct:.1f}%")
            all_hits[m]  += cat_hits[m]
            all_total[m] += cat_total[m]
        # delta: last model vs base
        base_pct  = cat_hits[model_names[0]] / max(1, cat_total[model_names[0]]) * 100
        last_pct  = cat_hits[model_names[-1]] / max(1, cat_total[model_names[-1]]) * 100
        sign = "+" if last_pct >= base_pct else ""
        row_cells.append(f"{sign}{last_pct - base_pct:.1f} pp")
        lines.append("| " + " | ".join(row_cells) + " |")

    # Overall row
    overall_cells = ["**Overall**", f"**{len(eval_rows)}**"]
    for m in model_names:
        pct = all_hits[m] / max(1, all_total[m]) * 100
        overall_cells.append(f"**{pct:.1f}%**")
    base_ov = all_hits[model_names[0]] / max(1, all_total[model_names[0]]) * 100
    last_ov = all_hits[model_names[-1]] / max(1, all_total[model_names[-1]]) * 100
    sign = "+" if last_ov >= base_ov else ""
    overall_cells.append(f"**{sign}{last_ov - base_ov:.1f} pp**")
    lines.append("| " + " | ".join(overall_cells) + " |")
    lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
