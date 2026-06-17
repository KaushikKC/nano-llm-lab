"""Read run_summary.json from all three training runs and print comparison table.

Usage:
    python scripts/compare_runs.py
"""
from __future__ import annotations

import json
from pathlib import Path


_RUNS = [
    ("Full FT (Stage 2)", "checkpoints/sft/run_summary.json"),
    ("LoRA (Stage 3)",    "checkpoints/lora/run_summary.json"),
    ("QLoRA (Stage 3)",   "checkpoints/qlora/run_summary.json"),
]

# Eval reports live in docs/, not in checkpoints/
_EVAL_REPORTS = {
    "Full FT (Stage 2)": "docs/sft/eval_report.md",
    "LoRA (Stage 3)":    "docs/stage3/eval_report.md",
    "QLoRA (Stage 3)":   "docs/stage3/eval_report.md",  # same report, QLoRA column
}

# Column index (0-based after splitting by "|") for the model's overall score.
# Overall row structure — sft: [cat, n, Base%, SFT%, Δ]; stage3: [cat, n, Base%, LoRA%, QLoRA%, Δ]
_EVAL_COL = {
    "Full FT (Stage 2)": 3,  # SFT% column in sft/eval_report.md
    "LoRA (Stage 3)":    3,  # LoRA% column in stage3/eval_report.md
    "QLoRA (Stage 3)":   4,  # QLoRA% column in stage3/eval_report.md (added after QLoRA eval)
}


def fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def estimated_memory_gb(summary: dict) -> str:
    mode = summary.get("mode", "full_ft")
    trainable = summary["trainable_params"]
    total = summary["total_params"]

    if mode == "qlora":
        # base in 4-bit + adapters in fp16
        base_gb = total * 0.5 / 1e9        # 4-bit = 0.5 bytes/param
        adapter_gb = trainable * 2 / 1e9   # fp16 = 2 bytes/param
        return f"~{base_gb + adapter_gb:.2f} GB"
    elif mode == "lora":
        # base in bf16 (frozen) + adapter grads in bf16 + Adam on adapters only
        base_gb = total * 2 / 1e9
        adapter_grad_gb = trainable * 2 / 1e9
        adam_gb = trainable * 8 / 1e9      # m + v in fp32
        return f"~{base_gb + adapter_grad_gb + adam_gb:.2f} GB"
    else:
        # full FT: weights + grads + Adam m+v (all in fp32 for Adam)
        weights_gb = total * 2 / 1e9       # bf16
        grads_gb   = total * 2 / 1e9
        adam_gb    = total * 8 / 1e9
        return f"~{weights_gb + grads_gb + adam_gb:.2f} GB"


def load_eval_score(label: str, col: int = 3) -> str:
    """col: 3=SFT/LoRA score, 4=QLoRA score in the stage3 report."""
    report_path = _EVAL_REPORTS.get(label)
    if not report_path:
        return "—"
    report = Path(report_path)
    if not report.exists():
        return "—"
    for line in report.read_text().splitlines():
        if "Overall" in line and "%" in line:
            parts = [p.strip().strip("*") for p in line.split("|") if p.strip()]
            if len(parts) > col:
                return parts[col]
    return "—"


def main() -> None:
    rows = []
    for label, path in _RUNS:
        p = Path(path)
        if not p.exists():
            rows.append((label, None))
            continue
        with open(p) as f:
            s = json.load(f)
        rows.append((label, s))

    # Header
    cols = ["Method", "Trainable params", "% of total", "Est. memory", "Wall time", "Eval score"]
    widths = [max(len(c), 22) for c in cols]

    def row_str(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print("\n" + "=" * 110)
    print("  nano-llm-lab — Stage 3 comparison: full FT vs LoRA vs QLoRA")
    print("=" * 110)
    print(row_str(cols))
    print("-" * 110)

    for label, s in rows:
        if s is None:
            print(row_str([label, "—", "—", "—", "—", "—"]))
            continue
        eval_score = load_eval_score(label, _EVAL_COL.get(label, 3))
        cells = [
            label,
            fmt_params(s["trainable_params"]),
            f"{s.get('trainable_pct', 100):.2f}%",
            estimated_memory_gb(s),
            f"{s['wall_time_min']:.1f} min",
            eval_score,
        ]
        print(row_str(cells))

    print("=" * 110)
    print()


if __name__ == "__main__":
    main()
