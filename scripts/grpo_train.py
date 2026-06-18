"""GRPO (Group Relative Policy Optimisation) training — Stage 5 Rung 3.

Implements the GRPO objective from DeepSeek-R1 (Shao et al. 2024):
  - Sample G responses per prompt (group rollout)
  - Score each with ternary reward (+1 correct / 0 abstain / -1 hallucination)
  - Normalise rewards within the group to get advantages
  - Update with clipped policy gradient + KL penalty against reference

IMPORTANT: Requires CUDA. MPS doesn't support the memory budget needed for
online rollouts (model + ref + G active sequences simultaneously).

Usage (on a CUDA machine):
    python scripts/grpo_train.py \
        --config configs/tool_grpo/qwen2.5-0.5b-tool-grpo.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanolab.grpo.config import GRPOConfig
from nanolab.grpo.reward import ternary_reward


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GRPODataset(Dataset):
    """Loads (system, prompt, expected_tool) triples from JSONL."""

    def __init__(self, path: str) -> None:
        self.rows = [
            json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": row["system"]},
            {"role": "user",   "content": row["prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _log_probs(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return sum of log-probs for non-masked label tokens."""
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids).logits  # [1, T, V]
    log_p = torch.log_softmax(logits[:, :-1], dim=-1)   # [1, T-1, V]
    tgt   = labels[:, 1:]                               # [1, T-1]
    mask  = tgt != -100
    gathered = log_p.gather(2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # [1,T-1]
    return (gathered * mask).sum(-1)  # [1]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: GRPOConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GRPO requires CUDA. Current device has no CUDA GPU.\n"
            "Run on a cloud GPU (A100/H100 recommended for 0.5B + G=4 rollouts).\n"
            "On MPS the rollout memory budget (model + ref + G sequences) exceeds 16GB."
        )

    device = torch.device("cuda")
    print(f"Device: {device}  |  group_size: {cfg.group_size}  |  lr: {cfg.lr}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(os.path.join(cfg.data_dir, "tool_names.json")) as f:
        valid_tool_names = set(json.load(f))

    print(f"Loading policy: {cfg.model_name}")
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    policy.train()

    print(f"Loading reference (CPU): {cfg.ref_model_name}")
    ref = AutoModelForCausalLM.from_pretrained(
        cfg.ref_model_name, torch_dtype=torch.float32
    )  # CPU
    ref.eval()

    train_ds = GRPODataset(os.path.join(cfg.data_dir, "train.jsonl"))
    loader   = DataLoader(train_ds, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)

    total_steps = len(loader) * cfg.epochs
    print(f"Dataset: {len(train_ds)} prompts | {total_steps} total steps")

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        for row in loader:
            row = {k: v[0] if isinstance(v, list) else v for k, v in row.items()}
            prompt_text    = build_prompt(tokenizer, row)
            expected_tool  = row["expected_tool"]

            # ----------------------------------------------------------------
            # Step 1: Sample G rollouts from the current policy
            # ----------------------------------------------------------------
            prompt_ids = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)

            with torch.no_grad():
                rollout_ids = policy.generate(
                    prompt_ids.expand(cfg.group_size, -1),
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=cfg.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

            responses = [
                tokenizer.decode(rollout_ids[i, prompt_ids.shape[1]:], skip_special_tokens=True)
                for i in range(cfg.group_size)
            ]

            # ----------------------------------------------------------------
            # Step 2: Compute ternary rewards and group-normalise → advantages
            # ----------------------------------------------------------------
            rewards = torch.tensor(
                ternary_reward(responses, [expected_tool] * cfg.group_size, valid_tool_names),
                dtype=torch.float32,
            )
            # GRPO: advantage = (r - mean(r)) / (std(r) + 1e-8)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # ----------------------------------------------------------------
            # Step 3: Policy gradient loss + KL penalty
            # ----------------------------------------------------------------
            policy_loss = torch.tensor(0.0, device=device)
            kl_total    = torch.tensor(0.0, device=device)

            for i in range(cfg.group_size):
                full_ids = rollout_ids[i:i+1]
                labels   = full_ids.clone()
                labels[:, :prompt_ids.shape[1]] = -100  # mask prompt

                pi_lp  = _log_probs(policy, full_ids, labels)
                with torch.no_grad():
                    ref_lp = _log_probs(ref, full_ids.cpu(), labels.cpu()).to(device)

                log_ratio = (pi_lp - ref_lp).clamp(-10.0, 10.0)
                ratio     = log_ratio.exp()

                a_i = adv[i].to(device)
                pg  = -torch.min(
                    ratio * a_i,
                    ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_i,
                )
                policy_loss = policy_loss + pg
                kl_total    = kl_total + log_ratio

            policy_loss = policy_loss / cfg.group_size
            kl_loss     = cfg.kl_coef * kl_total / cfg.group_size
            total_loss  = policy_loss + kl_loss

            if not torch.isnan(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                optimizer.step()
                torch.cuda.empty_cache()

            global_step += 1
            if global_step % cfg.log_interval == 0:
                mean_r = rewards.mean().item()
                print(f"epoch {epoch}/{cfg.epochs}  step {global_step}/{total_steps}  "
                      f"loss={total_loss.item():.4f}  mean_reward={mean_r:.3f}  "
                      f"kl={kl_loss.item():.4f}")

        # Save checkpoint
        ckpt_dir = Path(cfg.out_dir) / f"ckpt_epoch{epoch:03d}"
        policy.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  → checkpoint: {ckpt_dir}")

    out_hf = Path(cfg.out_dir) / "hf"
    policy.save_pretrained(out_hf)
    tokenizer.save_pretrained(out_hf)
    print(f"Policy saved to {out_hf}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cfg = GRPOConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    train(cfg)


if __name__ == "__main__":
    main()
