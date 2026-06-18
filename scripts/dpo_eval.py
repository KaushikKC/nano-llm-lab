"""Evaluate Base, SFT, and DPO models side-by-side on eval.jsonl.

Generates responses from all three models and scores them against keyword
rubrics. Also computes a win-rate: DPO vs SFT (keyword hits as proxy).

Usage:
    python scripts/dpo_eval.py \\
        --base   Qwen/Qwen2.5-0.5B \\
        --sft    checkpoints/sft/hf \\
        --dpo    checkpoints/dpo/hf \\
        --eval   data/dpo/eval.jsonl \\
        --out    docs/stage4/eval_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def build_prompt(row: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user",   "content": row["prompt"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(path: str, device: torch.device, dtype: torch.dtype):
    attn_impl = "eager" if str(device) == "mps" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation=attn_impl
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tokenizer, prompt: str, device: torch.device,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",  required=True)
    parser.add_argument("--sft",   required=True)
    parser.add_argument("--dpo",   required=True)
    parser.add_argument("--eval",  required=True)
    parser.add_argument("--out",   required=True)
    args = parser.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    dtype = torch.bfloat16
    print(f"Device: {device}")

    eval_rows = load_eval(args.eval)
    model_specs = [("Base", args.base), ("SFT", args.sft), ("DPO", args.dpo)]

    results: dict[str, list[dict]] = {}
    for label, path in model_specs:
        if not Path(path).exists():
            print(f"  {label}: skipped (path not found: {path})")
            continue
        print(f"\nLoading {label} …")
        model, tokenizer = load_model(path, device, dtype)
        rows_out = []
        for i, row in enumerate(eval_rows):
            kws = row.get("keywords", [])
            prompt = build_prompt(row, tokenizer)
            resp = generate(model, tokenizer, prompt, device)
            hits, total = keyword_score(resp, kws)
            print(f"  [{i+1}/{len(eval_rows)}] {row['category']}: {hits}/{total}")
            rows_out.append({
                "category": row["category"],
                "hits": hits, "total": total,
                "response": resp,
            })
        results[label] = rows_out
        del model

    _write_report(args.out, eval_rows, results)
    print(f"\nReport written to: {args.out}")


def _write_report(out_path: str, eval_rows: list[dict],
                  results: dict[str, list[dict]]) -> None:
    model_names = list(results.keys())
    categories = sorted({r["category"] for r in eval_rows})

    lines = ["# Stage 4 evaluation report — Base vs SFT vs DPO\n",
             "Keyword coverage per model. Win-rate = DPO rows where DPO hits ≥ SFT hits.\n"]

    lines.append("## Summary by category\n")
    header = ["Category", "n"] + model_names + [f"Δ (DPO − SFT)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    all_hits: dict[str, int] = {m: 0 for m in model_names}
    all_total: dict[str, int] = {m: 0 for m in model_names}

    for cat in categories:
        cat_hits  = {m: 0 for m in model_names}
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
        if "DPO" in results and "SFT" in results:
            d_pct = cat_hits["DPO"] / max(1, cat_total["DPO"]) * 100
            s_pct = cat_hits["SFT"] / max(1, cat_total["SFT"]) * 100
            sign = "+" if d_pct >= s_pct else ""
            row_cells.append(f"{sign}{d_pct - s_pct:.1f} pp")
        else:
            row_cells.append("—")
        lines.append("| " + " | ".join(row_cells) + " |")

    overall_cells = ["**Overall**", f"**{len(eval_rows)}**"]
    for m in model_names:
        pct = all_hits[m] / max(1, all_total[m]) * 100
        overall_cells.append(f"**{pct:.1f}%**")
    if "DPO" in results and "SFT" in results:
        d_ov = all_hits["DPO"] / max(1, all_total["DPO"]) * 100
        s_ov = all_hits["SFT"] / max(1, all_total["SFT"]) * 100
        sign = "+" if d_ov >= s_ov else ""
        overall_cells.append(f"**{sign}{d_ov - s_ov:.1f} pp**")
    else:
        overall_cells.append("—")
    lines.append("| " + " | ".join(overall_cells) + " |")
    lines.append("")

    # Win-rate section
    if "DPO" in results and "SFT" in results:
        lines.append("## Win-rate: DPO vs SFT\n")
        wins = draws = losses = 0
        for i in range(len(eval_rows)):
            dpo_h = results["DPO"][i]["hits"]
            sft_h = results["SFT"][i]["hits"]
            if dpo_h > sft_h:  wins += 1
            elif dpo_h == sft_h: draws += 1
            else: losses += 1
        n = len(eval_rows)
        lines.append(f"| Metric | Count | Rate |")
        lines.append(f"|---|---|---|")
        lines.append(f"| DPO wins (DPO > SFT) | {wins} | {wins/n:.1%} |")
        lines.append(f"| Draws (DPO = SFT)    | {draws} | {draws/n:.1%} |")
        lines.append(f"| DPO losses (DPO < SFT) | {losses} | {losses/n:.1%} |")
        lines.append(f"\n**Win-rate** (wins / total): **{wins/n:.1%}**  "
                     f"(excludes draws: {wins/(wins+losses):.1%})\n")

    # Per-example detail
    lines.append("## Per-example responses\n")
    for i, row in enumerate(eval_rows):
        lines.append(f"### Example {i+1} — `{row['category']}`\n")
        lines.append(f"**Prompt (first 200 chars)**: {row['prompt'][:200]}\n")
        lines.append(f"**Keywords**: {', '.join(row.get('keywords', []))}\n")
        lines.append("| Model | Hits | Score | Response (first 200 chars) |")
        lines.append("|---|---|---|---|")
        for m in model_names:
            r = results[m][i]
            pct = r["hits"] / max(1, r["total"]) * 100
            resp_snippet = r["response"][:200].replace("\n", " ")
            lines.append(f"| {m} | {r['hits']}/{r['total']} | {pct:.0f}% | {resp_snippet} |")
        lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
