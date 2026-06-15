"""Train a nanolab GPT model.

Usage:
    python scripts/train.py --config configs/small.yaml
    python scripts/train.py --config configs/tiny.yaml --max_steps 500
    python scripts/train.py --config configs/small.yaml --resume checkpoints/small/ckpt_last.pt
"""

import argparse
import math
import os
import time

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

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
    """AdamW with the standard decay/no-decay split: weight matrices (2D
    tensors) get weight decay, everything else (RMSNorm gains, the tied
    embedding/lm_head weight is 2D so it *is* decayed, biases if any) — in
    this model only RMSNorm weight vectors (1D) end up in the no-decay
    group."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=train_cfg.lr, betas=train_cfg.betas)


def get_lr(step: int, train_cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to `min_lr`."""
    if step < train_cfg.warmup_steps:
        return train_cfg.lr * (step + 1) / train_cfg.warmup_steps
    if step >= train_cfg.max_steps:
        return train_cfg.min_lr
    decay_ratio = (step - train_cfg.warmup_steps) / (train_cfg.max_steps - train_cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_cfg.min_lr + coeff * (train_cfg.lr - train_cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, dataset: TokenDataset, train_cfg: TrainConfig, device: torch.device
) -> float:
    model.eval()
    losses = torch.zeros(train_cfg.eval_iters)
    for i in range(train_cfg.eval_iters):
        x, y = dataset.get_batch(train_cfg.micro_batch_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


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
    writer = SummaryWriter(log_dir=os.path.join("runs", train_cfg.run_name))

    tokens_per_step = train_cfg.micro_batch_size * train_cfg.grad_accum_steps * model_cfg.max_seq_len
    print(f"Tokens per optimizer step: {tokens_per_step:,}")

    model.train()
    t0 = time.time()
    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_cfg.grad_accum_steps):
            x, y = train_ds.get_batch(train_cfg.micro_batch_size, device)
            _, loss = model(x, y)
            (loss / train_cfg.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.log_interval == 0:
            elapsed = time.time() - t0
            tok_per_sec = tokens_per_step * (step - start_step + 1) / max(elapsed, 1e-8)
            print(
                f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} "
                f"| {tok_per_sec:,.0f} tok/s"
            )
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/tokens_per_sec", tok_per_sec, step)

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            val_loss = estimate_loss(model, val_ds, train_cfg, device)
            train_loss = estimate_loss(model, train_ds, train_cfg, device)
            print(f"step {step:6d} | eval: train loss {train_loss:.4f} | val loss {val_loss:.4f}")
            writer.add_scalar("eval/train_loss", train_loss, step)
            writer.add_scalar("eval/val_loss", val_loss, step)


if __name__ == "__main__":
    main()
