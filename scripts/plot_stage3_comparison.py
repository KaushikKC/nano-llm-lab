"""Plot val-loss curves for Full FT, LoRA, and QLoRA on one axes.

Usage:
    python scripts/plot_stage3_comparison.py --out docs/images/stage3_loss_comparison.png
    python scripts/plot_stage3_comparison.py --out docs/images/stage3_loss_comparison.png --qlora runs/qlora_qwen0.5b
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


_RUNS = {
    "Full FT": "runs/sft_qwen0.5b",
    "LoRA":    "runs/lora_qwen0.5b",
}


def load_scalar(run_dir: str, tag: str) -> tuple[list[int], list[float]]:
    acc = EventAccumulator(run_dir)
    acc.Reload()
    events = acc.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--qlora", default=None, help="QLoRA tensorboard run dir (optional)")
    args = parser.parse_args()

    runs = dict(_RUNS)
    if args.qlora and Path(args.qlora).exists():
        runs["QLoRA"] = args.qlora

    colors = {"Full FT": "#e05c5c", "LoRA": "#4a90d9", "QLoRA": "#5cb85c"}
    styles = {"Full FT": "-", "LoRA": "--", "QLoRA": "-."}

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, run_dir in runs.items():
        if not Path(run_dir).exists():
            continue
        try:
            v_steps, v_loss = load_scalar(run_dir, "eval/val_loss")
        except Exception as e:
            print(f"  {label}: skipped ({e})")
            continue
        c = colors.get(label, "gray")
        s = styles.get(label, "-")
        ax.plot(v_steps, v_loss, label=label, color=c, linestyle=s, linewidth=1.8)

    ax.set_xlabel("optimizer step (1 step = 1 epoch)")
    ax.set_ylabel("val loss")
    ax.set_title("Validation loss — Full FT vs LoRA vs QLoRA")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Stage 3 — all methods trained on same Solidity/DeFi dataset, 10 epochs", fontsize=11)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
