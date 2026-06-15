"""Train a nanolab GPT model.

Usage:
    python scripts/train.py --config configs/small.yaml
    python scripts/train.py --config configs/tiny.yaml --max_steps 500
    python scripts/train.py --config configs/small.yaml --resume checkpoints/small/ckpt_last.pt
"""

import argparse
import os

import torch
import yaml

from nanolab.config import GPTConfig, TrainConfig
from nanolab.data.dataset import TokenDataset
from nanolab.model.gpt import GPT
from nanolab.utils import get_device, set_seed


def load_config(path: str) -> tuple[GPTConfig, TrainConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    model_cfg = GPTConfig(**raw.get("model", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))
    return model_cfg, train_cfg


def build_optimizer(model: torch.nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    model_cfg, train_cfg = load_config(args.config)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps

    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)
    print(f"Using device: {device}")

    train_ds = TokenDataset(os.path.join(train_cfg.data_dir, "train.bin"), model_cfg.max_seq_len)
    val_ds = TokenDataset(os.path.join(train_cfg.data_dir, "val.bin"), model_cfg.max_seq_len)

    model = GPT(model_cfg).to(device)
    print(
        f"Model: {model.num_params():,} params "
        f"({model.num_params(non_embedding=True):,} non-embedding)"
    )

    optimizer = build_optimizer(model, train_cfg)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from {args.resume} at step {start_step}")

    os.makedirs(train_cfg.out_dir, exist_ok=True)


if __name__ == "__main__":
    main()
