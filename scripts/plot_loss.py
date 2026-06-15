"""Plot train/val loss curves from a tensorboard run directory.

Usage:
    python scripts/plot_loss.py --run runs/small --out docs/images/small_loss.png
"""

import argparse
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(run_dir: str, tag: str) -> tuple[list[int], list[float]]:
    acc = EventAccumulator(run_dir)
    acc.Reload()
    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="tensorboard run directory")
    parser.add_argument("--out", required=True, help="output image path")
    args = parser.parse_args()

    train_steps, train_loss = load_scalar(args.run, "eval/train_loss")
    val_steps, val_loss = load_scalar(args.run, "eval/val_loss")

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_loss, label="train loss")
    plt.plot(val_steps, val_loss, label="val loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(os.path.basename(args.run.rstrip("/")) + " — loss")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
