"""LoRA / QLoRA fine-tuning script for Stage 3.

Usage:
    # LoRA (MPS / CUDA)
    python scripts/lora_train.py --config configs/lora/qwen2.5-0.5b-lora.yaml

    # QLoRA (4-bit base; CUDA-accelerated; CPU fallback on MPS)
    python scripts/lora_train.py --config configs/lora/qwen2.5-0.5b-qlora.yaml --qlora
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanolab.peft.config import LoraTrainConfig, QLoraTrainConfig
from nanolab.sft.dataset import SFTDataset, collate_fn


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str, qlora: bool) -> LoraTrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg_cls = QLoraTrainConfig if qlora else LoraTrainConfig
    cfg = cfg_cls()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# LR schedule (same cosine+warmup shape as Stage 1 and 2)
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: LoraTrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


# ---------------------------------------------------------------------------
# Device + dtype resolution
# ---------------------------------------------------------------------------

def resolve_device(cfg: LoraTrainConfig, qlora: bool) -> torch.device:
    if cfg.device == "auto":
        if qlora:
            # bitsandbytes 4-bit kernels are CUDA-only; fall back to CPU on MPS
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("QLoRA: bitsandbytes 4-bit requires CUDA — falling back to CPU.")
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


# ---------------------------------------------------------------------------
# Model loading: plain LoRA or QLoRA
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: LoraTrainConfig, device: torch.device, qlora: bool):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    adapter_dtype = dtype_map[cfg.dtype]

    # MPS attention fix (same as Stage 2)
    attn_impl = "eager" if str(device) == "mps" else "sdpa"

    if qlora:
        from transformers import BitsAndBytesConfig
        bnb_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_dtype_map[cfg.bnb_4bit_compute_dtype],
        )
        print(f"Loading model in 4-bit (NF4) for QLoRA …")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_cfg,
            attn_implementation=attn_impl,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"Loading model: {cfg.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=adapter_dtype,
            attn_implementation=attn_impl,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable "
          f"({trainable/total*100:.2f}%)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: LoraTrainConfig, device: torch.device, qlora: bool) -> None:
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(cfg, device, qlora)

    # Datasets
    train_ds = SFTDataset(
        os.path.join(cfg.data_dir, "train.jsonl"), tokenizer, cfg.max_seq_len
    )
    val_ds = SFTDataset(
        os.path.join(cfg.data_dir, "val.jsonl"), tokenizer, cfg.max_seq_len
    )
    pad_id = tokenizer.pad_token_id
    _collate = partial(collate_fn, pad_id=pad_id, fixed_len=cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.micro_batch_size, shuffle=True,
        collate_fn=_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.micro_batch_size, shuffle=False,
        collate_fn=_collate, drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps))
    total_steps = steps_per_epoch * cfg.epochs

    writer = SummaryWriter(log_dir=f"runs/{cfg.run_name}")
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val | "
          f"{steps_per_epoch} optimizer steps/epoch × {cfg.epochs} epochs = {total_steps} total")
    print(f"Tensorboard: runs/{cfg.run_name}")

    # MPS warmup: compile forward+backward Metal kernels before the epoch loop
    if str(device) == "mps":
        print("Warming up MPS kernels (forward + backward)…")
        _ids = torch.ones(cfg.micro_batch_size, cfg.max_seq_len, dtype=torch.long, device=device)
        _out = model(_ids, labels=_ids)
        _out.loss.backward()
        torch.mps.synchronize()
        model.zero_grad()
        del _ids, _out
        print("Warmup done.")

    global_step = 0
    train_loss = float("nan")
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        accum_loss = 0.0
        micro_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            if outputs.loss.isnan():
                micro_steps += 1
                continue

            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()   # accumulate divided loss (matches Stage 2)
            micro_steps += 1

            if micro_steps % cfg.grad_accum_steps == 0 or batch_idx == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                lr_now = get_lr(global_step, cfg, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                train_loss = accum_loss * cfg.grad_accum_steps / max(1, micro_steps)
                accum_loss = 0.0
                micro_steps = 0

                writer.add_scalar("train/loss", train_loss, global_step)
                writer.add_scalar("train/lr", lr_now, global_step)

                if global_step % cfg.log_interval == 0:
                    elapsed = int(time.time() - t_start)
                    print(f"epoch {epoch}/{cfg.epochs}  step {global_step}/{total_steps}  "
                          f"train_loss={train_loss:.4f}  lr={lr_now:.2e}  {elapsed}s")

        # Epoch-end validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, labels=labels)
                if not out.loss.isnan():
                    val_losses.append(out.loss.item())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        writer.add_scalar("eval/train_loss", train_loss, global_step)
        writer.add_scalar("eval/val_loss", val_loss, global_step)
        print(f"  → epoch {epoch} val_loss={val_loss:.4f}")

        # Save adapter checkpoint each epoch
        ckpt_path = out_dir / f"adapter_epoch{epoch:03d}"
        model.save_pretrained(str(ckpt_path))
        print(f"  → adapter saved: {ckpt_path.name}")

    writer.close()
    elapsed_min = (time.time() - t_start) / 60

    # Save final adapter in HF format
    hf_out = str(out_dir / "hf")
    model.save_pretrained(hf_out)
    tokenizer.save_pretrained(hf_out)
    print(f"Adapter saved to {hf_out}")

    # Run summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    summary = {
        "model_name": cfg.model_name,
        "mode": "qlora" if qlora else "lora",
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(trainable / total * 100, 3),
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "epochs": cfg.epochs,
        "total_optimizer_steps": global_step,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "effective_batch_size": cfg.micro_batch_size * cfg.grad_accum_steps,
        "lr": cfg.lr,
        "device": str(device),
        "wall_time_sec": round(time.time() - t_start, 1),
        "wall_time_min": round(elapsed_min, 1),
        "hardware": "Apple M3 16GB / MPS" if str(device) == "mps" else str(device),
        "cost_usd": 0,
    }
    summary_path = str(out_dir / "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Run summary: {summary_path}")
    print(f"Done — {elapsed_min:.1f} min total")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.qlora)
    device = resolve_device(cfg, args.qlora)
    print(f"Device: {device}  |  mode: {'qlora' if args.qlora else 'lora'}")
    train(cfg, device, args.qlora)
