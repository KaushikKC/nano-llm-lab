"""DPO (Direct Preference Optimization) training — Stage 4.

Trains a policy model starting from the SFT checkpoint using preference pairs
(chosen / rejected responses). A frozen copy of the SFT model serves as the
reference to compute log-ratio penalties.

DPO loss (Rafailov et al. 2023):
    L = -E[ log σ( β * (log π_θ(y_w|x)/π_ref(y_w|x)
                       - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]

Usage:
    python scripts/dpo_train.py --config configs/dpo/qwen2.5-0.5b-dpo.yaml
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanolab.dpo.config import DPOTrainConfig
from nanolab.dpo.dataset import DPODataset, dpo_collate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> DPOTrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = DPOTrainConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# LR schedule (cosine + warmup)
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: DPOTrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(cfg: DPOTrainConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-token log-probability
# ---------------------------------------------------------------------------

def _log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of per-token log-probs for non-masked positions.

    logits: (B, T, V)
    labels: (B, T)  — -100 marks positions to ignore
    returns: (B,)
    """
    # Keep native dtype (bf16/fp32) — avoid .float() cast to prevent large MPS allocs
    # on the 151k-vocab Qwen tokenizer (510 × 151936 × 4B = 310 MB per call in fp32)
    shifted = logits[:, :-1, :]                                    # (B, T-1, V)
    log_probs = F.log_softmax(shifted, dim=-1)
    tgt = labels[:, 1:].clone()                                    # (B, T-1)
    mask = tgt != -100
    tgt[~mask] = 0

    gathered = log_probs.gather(2, tgt.unsqueeze(2)).squeeze(2)    # (B, T-1)
    return (gathered * mask.float()).sum(dim=-1)                    # (B,)


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def dpo_loss(
    policy_model,
    ref_model,
    batch: dict,
    beta: float,
    device: torch.device,
    ref_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, implicit_reward_margin).

    ref_device may differ from device (e.g. CPU) to save GPU memory.
    implicit_reward_margin > 0 means policy prefers chosen over rejected.
    """
    ids_w = batch["input_ids_w"].to(device)
    ids_l = batch["input_ids_l"].to(device)
    lbl_w = batch["labels_w"].to(device)
    lbl_l = batch["labels_l"].to(device)

    # Policy log-probs (on GPU/MPS)
    policy_logits_w = policy_model(input_ids=ids_w).logits
    policy_logits_l = policy_model(input_ids=ids_l).logits
    pi_lp_w = _log_probs_from_logits(policy_logits_w, lbl_w)
    pi_lp_l = _log_probs_from_logits(policy_logits_l, lbl_l)

    # Reference log-probs (on ref_device — CPU to save MPS memory)
    with torch.no_grad():
        ref_logits_w = ref_model(input_ids=ids_w.to(ref_device)).logits
        ref_logits_l = ref_model(input_ids=ids_l.to(ref_device)).logits
    ref_lp_w = _log_probs_from_logits(ref_logits_w, lbl_w.to(ref_device)).to(device)
    ref_lp_l = _log_probs_from_logits(ref_logits_l, lbl_l.to(ref_device)).to(device)

    # DPO logits: β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))
    pi_log_ratio_w = pi_lp_w - ref_lp_w
    pi_log_ratio_l = pi_lp_l - ref_lp_l
    logits_dpo = beta * (pi_log_ratio_w - pi_log_ratio_l)

    loss = -F.logsigmoid(logits_dpo).mean()
    reward_margin = logits_dpo.detach().mean()  # positive = policy prefers chosen
    return loss, reward_margin


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: DPOTrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[cfg.dtype]
    attn_impl = "eager" if str(device) == "mps" else "sdpa"

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  beta: {cfg.beta}  |  lr: {cfg.lr}")

    # Load policy model (trainable)
    print(f"Loading policy: {cfg.model_name}")
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, attn_implementation=attn_impl
    ).to(device)
    policy.gradient_checkpointing_enable()
    policy.train()

    # Load reference model on CPU — keeps it off MPS to avoid OOM with two 494M models
    ref_device = torch.device("cpu")
    print(f"Loading reference on CPU (saves MPS memory): {cfg.ref_model_name}")
    ref = AutoModelForCausalLM.from_pretrained(
        cfg.ref_model_name, torch_dtype=torch.float32  # CPU prefers float32
    ).to(ref_device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_ds = DPODataset(os.path.join(cfg.data_dir, "train.jsonl"), tokenizer, cfg.max_seq_len)
    val_ds   = DPODataset(os.path.join(cfg.data_dir, "val.jsonl"),   tokenizer, cfg.max_seq_len)

    _collate = partial(dpo_collate, pad_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.micro_batch_size, shuffle=False, collate_fn=_collate)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps))
    total_steps = steps_per_epoch * cfg.epochs
    writer = SummaryWriter(log_dir=f"runs/{cfg.run_name}")

    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val | "
          f"{steps_per_epoch} opt steps/epoch × {cfg.epochs} = {total_steps} total")

    # MPS warmup
    if str(device) == "mps":
        print("Warming up MPS kernels…")
        _dummy = torch.ones(1, 4, dtype=torch.long, device=device)
        policy(_dummy).logits.sum().backward()
        torch.mps.synchronize()
        policy.zero_grad()

    global_step = 0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        policy.train()
        accum_loss = 0.0
        accum_margin = 0.0
        micro_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            loss, margin = dpo_loss(policy, ref, batch, cfg.beta, device, ref_device)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss   += loss.item()
            accum_margin += margin.item()
            micro_steps  += 1

            if micro_steps % cfg.grad_accum_steps == 0 or batch_idx == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                lr_now = get_lr(global_step, cfg, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if str(device) == "mps":
                    torch.mps.empty_cache()

                train_loss   = accum_loss   * cfg.grad_accum_steps / max(1, micro_steps)
                train_margin = accum_margin / max(1, micro_steps // cfg.grad_accum_steps)
                accum_loss   = 0.0
                accum_margin = 0.0
                micro_steps  = 0

                writer.add_scalar("train/loss",          train_loss,   global_step)
                writer.add_scalar("train/reward_margin", train_margin, global_step)
                writer.add_scalar("train/lr",            lr_now,       global_step)

                if global_step % cfg.log_interval == 0:
                    elapsed = int(time.time() - t_start)
                    print(f"epoch {epoch}/{cfg.epochs}  step {global_step}/{total_steps}  "
                          f"loss={train_loss:.4f}  margin={train_margin:.4f}  "
                          f"lr={lr_now:.2e}  {elapsed}s")

        # Epoch-end validation
        policy.eval()
        val_losses, val_margins, val_accs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                l, m = dpo_loss(policy, ref, batch, cfg.beta, device, ref_device)
                val_losses.append(l.item())
                val_margins.append(m.item())
                val_accs.append(float(m > 0))

        val_loss   = sum(val_losses)  / len(val_losses)  if val_losses  else float("nan")
        val_margin = sum(val_margins) / len(val_margins) if val_margins else float("nan")
        val_acc    = sum(val_accs)    / len(val_accs)    if val_accs    else float("nan")
        writer.add_scalar("eval/val_loss",   val_loss,   global_step)
        writer.add_scalar("eval/val_margin", val_margin, global_step)
        writer.add_scalar("eval/val_acc",    val_acc,    global_step)

        print(f"  → epoch {epoch}  val_loss={val_loss:.4f}  margin={val_margin:.4f}  "
              f"acc={val_acc:.1%}  (acc>0.5 means policy prefers chosen over rejected)")

        ckpt = out_dir / f"ckpt_epoch{epoch:03d}"
        policy.save_pretrained(str(ckpt))
        print(f"  → checkpoint: {ckpt.name}")

    writer.close()
    elapsed_min = (time.time() - t_start) / 60

    hf_out = str(out_dir / "hf")
    policy.save_pretrained(hf_out)
    tokenizer.save_pretrained(hf_out)
    print(f"Policy saved to {hf_out}")

    summary = {
        "model_name": cfg.model_name,
        "mode": "dpo",
        "beta": cfg.beta,
        "epochs": cfg.epochs,
        "total_optimizer_steps": global_step,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "lr": cfg.lr,
        "device": str(device),
        "wall_time_sec": round(time.time() - t_start, 1),
        "wall_time_min": round(elapsed_min, 1),
        "hardware": "Apple M3 16GB / MPS" if str(device) == "mps" else str(device),
        "cost_usd": 0,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done — {elapsed_min:.1f} min")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
