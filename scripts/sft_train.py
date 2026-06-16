"""Stage 2 SFT training script.

Loads a pretrained Hugging Face causal LM (default: Qwen/Qwen2.5-0.5B),
fine-tunes it on the hand-built Solidity/DeFi JSONL dataset with:
  - prompt-loss masking (only assistant tokens contribute to the loss)
  - AdamW + cosine-with-warmup LR schedule
  - gradient accumulation + gradient clipping
  - tensorboard logging (same tags as Stage 1 so plot_loss.py is reusable)
  - checkpoint save/resume + run_summary.json

Usage:
    python scripts/sft_train.py
    python scripts/sft_train.py --config configs/sft/qwen2.5-0.5b.yaml
    python scripts/sft_train.py --config configs/sft/qwen2.5-0.5b.yaml --resume checkpoints/sft/ckpt_last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

# project root on sys.path via `pip install -e .`
from nanolab.sft.config import SFTConfig
from nanolab.sft.dataset import SFTDataset, collate_fn


# ─── device / dtype helpers ───────────────────────────────────────────────────

def get_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


# ─── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: SFTConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    decay_steps = total_steps - cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


# ─── config loader ────────────────────────────────────────────────────────────

def load_config(yaml_path: str | None) -> SFTConfig:
    cfg = SFTConfig()
    if yaml_path:
        with open(yaml_path) as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


# ─── training ─────────────────────────────────────────────────────────────────

def train(cfg: SFTConfig, resume_path: str | None = None) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    dtype = get_torch_dtype(cfg.dtype)

    # ── model + tokenizer ──────────────────────────────────────────────────────
    print(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
    ).to(device)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── datasets + loaders ────────────────────────────────────────────────────
    data_dir = Path(cfg.data_dir)
    train_ds = SFTDataset(data_dir / "train.jsonl", tokenizer, cfg.max_seq_len)
    val_ds   = SFTDataset(data_dir / "val.jsonl",   tokenizer, cfg.max_seq_len)

    pad_id = tokenizer.pad_token_id
    _collate = partial(collate_fn, pad_id=pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    steps_per_epoch = math.ceil(len(train_ds) / cfg.micro_batch_size / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val | "
          f"{steps_per_epoch} optimizer steps/epoch × {cfg.epochs} epochs = {total_steps} total")

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        weight_decay=cfg.weight_decay,
    )

    # ── checkpoint resume ─────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if resume_path:
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)

    # ── logging ───────────────────────────────────────────────────────────────
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path("runs") / cfg.run_name
    writer = SummaryWriter(str(run_dir))
    print(f"Tensorboard: {run_dir}")

    t0 = time.time()

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0
        micro_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            # forward
            outputs = model(input_ids=input_ids, labels=labels)
            if outputs.loss.isnan():
                # All labels are -100 (prompt-only after truncation) — skip batch.
                micro_steps += 1
                continue
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
            micro_steps += 1

            is_update_step = (micro_steps % cfg.grad_accum_steps == 0) or (
                batch_idx == len(train_loader) - 1
            )

            if is_update_step:
                # LR update
                lr_now = get_lr(global_step, cfg, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

                # gradient clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % cfg.log_interval == 0:
                    elapsed = time.time() - t0
                    train_loss = accum_loss * cfg.grad_accum_steps / micro_steps
                    print(
                        f"epoch {epoch+1}/{cfg.epochs}  step {global_step}/{total_steps}"
                        f"  train_loss={train_loss:.4f}  lr={lr_now:.2e}  {elapsed:.0f}s"
                    )
                    writer.add_scalar("train/loss", train_loss, global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)

                accum_loss = 0.0
                micro_steps = 0

        # ── end-of-epoch eval ─────────────────────────────────────────────────
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, labels=labels)
                val_losses.append(outputs.loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"  → epoch {epoch+1} val_loss={val_loss:.4f}")
        writer.add_scalar("eval/val_loss", val_loss, global_step)

        # Also record the last train loss at epoch boundaries under the same
        # "eval/train_loss" tag used by Stage 1 so plot_loss.py works unchanged.
        if train_loader.dataset:
            writer.add_scalar("eval/train_loss", train_loss, global_step)

        # ── checkpoint ────────────────────────────────────────────────────────
        ckpt_path = out_dir / f"ckpt_epoch{epoch+1:03d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_loss,
                "config": asdict(cfg),
            },
            ckpt_path,
        )
        # rolling last pointer
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_loss,
                "config": asdict(cfg),
            },
            out_dir / "ckpt_last.pt",
        )
        print(f"  → checkpoint saved: {ckpt_path.name}")

    # ── save HF-format model for inference ────────────────────────────────────
    hf_out = out_dir / "hf"
    model.save_pretrained(hf_out)
    tokenizer.save_pretrained(hf_out)
    print(f"HF model saved to {hf_out}")

    # ── run summary ───────────────────────────────────────────────────────────
    wall_time = time.time() - t0
    summary = {
        "model_name": cfg.model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs": cfg.epochs,
        "total_optimizer_steps": global_step,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "effective_batch_size": cfg.micro_batch_size * cfg.grad_accum_steps,
        "lr": cfg.lr,
        "wall_time_sec": round(wall_time, 1),
        "wall_time_min": round(wall_time / 60, 1),
        "hardware": "Apple M3 16GB / MPS" if str(device) == "mps" else str(device),
        "cost_usd": 0,
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Run summary: {summary_path}")
    print(f"Done — {wall_time/60:.1f} min total")

    writer.close()


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 SFT training")
    parser.add_argument("--config", default=None, help="Path to YAML config (overrides defaults)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
