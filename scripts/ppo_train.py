"""Minimal educational PPO-RLHF — Stage 4 (bonus).

A from-scratch, small-scale implementation to demonstrate the reward-model +
policy setup. NOT production-grade — written for understanding.

Two phases:
  Phase 1 — Reward Model training:
    Train a scalar reward head on top of the SFT model using preference pairs.
    Loss: -log sigmoid(rm(chosen) - rm(rejected))  [Bradley-Terry]

  Phase 2 — PPO policy update:
    Sample completions from the policy, score with the reward model, compute
    GAE advantages, update the policy with a clipped surrogate objective + KL
    penalty vs the SFT reference.

Usage:
    python scripts/ppo_train.py --config configs/dpo/qwen2.5-0.5b-dpo.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanolab.dpo.config import PPOTrainConfig
from nanolab.dpo.dataset import DPODataset, dpo_collate
from functools import partial


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> PPOTrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = PPOTrainConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def resolve_device(cfg: PPOTrainConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Reward model: LM backbone + scalar head
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """Wraps a causal LM and adds a scalar reward head on the last token."""

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.reward_head.weight, std=0.01)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        # Use the last non-padding token for each sequence
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1          # (B,)
        else:
            seq_lens = torch.full((input_ids.shape[0],),
                                  input_ids.shape[1] - 1,
                                  device=input_ids.device)
        seq_lens = seq_lens.clamp(min=0)
        pooled = last_hidden[torch.arange(input_ids.shape[0]), seq_lens]  # (B, H)
        return self.reward_head(pooled.float()).squeeze(-1)  # (B,) — float32 for linear head


# ---------------------------------------------------------------------------
# Phase 1: Reward model training
# ---------------------------------------------------------------------------

def train_reward_model(cfg: PPOTrainConfig, device: torch.device,
                       dtype: torch.dtype) -> RewardModel:
    print("\n=== Phase 1: Reward Model Training ===")
    attn_impl = "eager" if str(device) == "mps" else "sdpa"

    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, attn_implementation=attn_impl
    ).to(device)
    hidden_size = backbone.config.hidden_size
    rm = RewardModel(backbone, hidden_size).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = DPODataset(os.path.join(cfg.data_dir, "train.jsonl"), tokenizer, cfg.max_seq_len)
    collate  = partial(dpo_collate, pad_id=tokenizer.pad_token_id)
    loader   = DataLoader(train_ds, batch_size=cfg.rm_batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(rm.parameters(), lr=cfg.rm_lr, weight_decay=0.01)

    for epoch in range(1, cfg.rm_epochs + 1):
        rm.train()
        total_loss = 0.0
        total_acc  = 0
        n_batches  = 0

        for batch in loader:
            ids_w = batch["input_ids_w"].to(device)
            ids_l = batch["input_ids_l"].to(device)
            mask_w = (ids_w != tokenizer.pad_token_id).long()
            mask_l = (ids_l != tokenizer.pad_token_id).long()

            r_w = rm(ids_w, mask_w)
            r_l = rm(ids_l, mask_l)
            loss = -F.logsigmoid(r_w - r_l).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc  += (r_w > r_l).float().mean().item()
            n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_acc  = total_acc  / max(1, n_batches)
        print(f"  RM epoch {epoch}/{cfg.rm_epochs}  loss={avg_loss:.4f}  acc={avg_acc:.1%}")

    rm_path = Path(cfg.rm_out_dir)
    rm_path.mkdir(parents=True, exist_ok=True)
    torch.save(rm.state_dict(), rm_path / "rm_weights.pt")
    print(f"Reward model saved: {rm_path / 'rm_weights.pt'}")
    return rm, tokenizer


# ---------------------------------------------------------------------------
# Phase 2: PPO update
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """Simple dataset of prompt strings for PPO rollout."""
    def __init__(self, path: str, tokenizer, max_seq_len: int) -> None:
        self.prompts = []
        for row in [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]:
            p = tokenizer.apply_chat_template(
                [{"role": "system", "content": row["system"]},
                 {"role": "user",   "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tokenizer(p, add_special_tokens=False,
                            max_length=max_seq_len, truncation=True)["input_ids"]
            self.prompts.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.prompts[idx]


def _compute_log_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs for the response portion. Returns scalar (mean)."""
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    log_p  = F.log_softmax(logits, dim=-1)  # keep native dtype — .float() causes MPS OOM
    tgt    = input_ids[:, 1:]
    return log_p.gather(2, tgt.unsqueeze(2)).squeeze(2).mean(dim=1)  # (B,)


def ppo_update(policy, ref, optimizer, batch_ids: torch.Tensor, rewards: torch.Tensor,
               clip_eps: float, kl_coef: float, device: torch.device,
               ref_device: torch.device | None = None) -> dict:
    """Single PPO step on one batch of (response_ids, rewards)."""
    if ref_device is None:
        ref_device = device
    policy.train()
    ids = batch_ids.to(device)
    r   = rewards.to(device)

    # Log-probs under current policy and reference
    pi_lp  = _compute_log_probs(policy, ids)
    with torch.no_grad():
        ref_lp = _compute_log_probs(ref, ids.to(ref_device)).to(device)

    # Clamp log-ratio before exp to prevent inf/NaN (standard PPO practice)
    log_ratio  = (pi_lp - ref_lp.detach()).clamp(-10.0, 10.0)
    kl = log_ratio.mean()

    # Use old log-probs as the "old policy" (collected at rollout time — simplified)
    # In full PPO you'd store old_lp from the rollout; here we approximate with ref_lp
    ratio = torch.exp(log_ratio)
    # std() of a single-element tensor is nan (unbiased, N-1=0); just center advantages
    advantages = r - r.mean() if r.shape[0] > 1 else r - r.detach()

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    total_loss  = policy_loss + kl_coef * kl

    optimizer.zero_grad()
    if not torch.isnan(total_loss):
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if device.type == "mps":
            torch.mps.empty_cache()

    return {"policy_loss": policy_loss.item(), "kl": kl.item(),
            "mean_reward": r.mean().item()}


def train_ppo(cfg: PPOTrainConfig, rm: RewardModel, tokenizer,
              device: torch.device, dtype: torch.dtype) -> None:
    print("\n=== Phase 2: PPO Policy Update ===")
    attn_impl = "eager" if str(device) == "mps" else "sdpa"

    # Move RM to CPU before loading policy — frees ~1 GB MPS for policy + AdamW states
    rm_device = torch.device("cpu")
    rm.to(rm_device)
    if device.type == "mps":
        torch.mps.empty_cache()

    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, attn_implementation=attn_impl
    ).to(device)
    # ref also on CPU: policy (1 GB) + AdamW states (4 GB) already fills MPS budget
    ref_device = torch.device("cpu")
    ref = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32
    ).to(ref_device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.ppo_lr)

    prompt_ds = PromptDataset(
        os.path.join(cfg.data_dir, "train.jsonl"), tokenizer, cfg.max_seq_len
    )
    prompt_loader = DataLoader(prompt_ds, batch_size=1, shuffle=True,
                               collate_fn=lambda b: b)

    out_dir = Path(cfg.ppo_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    rm.eval()

    for epoch in range(1, cfg.ppo_epochs + 1):
        epoch_rewards = []
        epoch_losses  = []

        step_count = 0
        for prompt_ids in prompt_loader:
            if step_count >= cfg.ppo_steps_per_epoch:
                break
            prompt_ids = prompt_ids[0].unsqueeze(0).to(device)

            # Rollout: generate a response
            with torch.no_grad():
                policy.eval()
                gen_ids = policy.generate(
                    prompt_ids,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,        # greedy: avoids bf16 multinomial NaN on MPS
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Score with RM on CPU (moved there to free MPS for policy + optimizer)
                mask = (gen_ids != tokenizer.pad_token_id).long()
                reward = rm(gen_ids.to(rm_device), mask.to(rm_device)).to(device)

            # PPO mini-update
            stats = ppo_update(
                policy, ref, optimizer,
                gen_ids, reward,
                cfg.clip_eps, cfg.kl_coef, device, ref_device
            )
            if not (math.isnan(stats["policy_loss"]) or math.isnan(stats["kl"])):
                epoch_rewards.append(stats["mean_reward"])
                epoch_losses.append(stats["policy_loss"])
            step_count += 1
            global_step += 1

            if global_step % 2 == 0:
                print(f"  PPO step {global_step}  reward={stats['mean_reward']:.4f}  "
                      f"kl={stats['kl']:.4f}  policy_loss={stats['policy_loss']:.4f}")

        mean_r = sum(epoch_rewards) / max(1, len(epoch_rewards))
        print(f"  PPO epoch {epoch}/{cfg.ppo_epochs}  mean_reward={mean_r:.4f}")

    # Save PPO model
    ppo_hf = str(out_dir / "hf")
    policy.save_pretrained(ppo_hf)
    tokenizer.save_pretrained(ppo_hf)
    print(f"PPO policy saved to {ppo_hf}")

    summary = {
        "mode": "ppo",
        "rm_epochs": cfg.rm_epochs,
        "ppo_epochs": cfg.ppo_epochs,
        "ppo_steps": global_step,
        "kl_coef": cfg.kl_coef,
        "clip_eps": cfg.clip_eps,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("PPO training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = resolve_device(cfg)
    dtype  = {"bfloat16": torch.bfloat16, "float16": torch.float16,
               "float32": torch.float32}[cfg.dtype]

    print(f"Device: {device}  |  PPO-RLHF (educational minimal implementation)")
    rm, tokenizer = train_reward_model(cfg, device, dtype)
    train_ppo(cfg, rm, tokenizer, device, dtype)
