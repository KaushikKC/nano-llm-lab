from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DPOTrainConfig:
    model_name: str = "checkpoints/sft/hf"   # start from SFT checkpoint
    ref_model_name: str = "checkpoints/sft/hf"  # frozen reference
    data_dir: str = "data/dpo"
    out_dir: str = "checkpoints/dpo"
    run_name: str = "dpo_qwen0.5b"
    max_seq_len: int = 512
    beta: float = 0.1               # KL penalty weight
    micro_batch_size: int = 1       # DPO loads 2 sequences per example (chosen+rejected)
    grad_accum_steps: int = 4
    epochs: int = 3
    lr: float = 5.0e-7              # DPO uses much smaller LR than SFT
    min_lr: float = 5.0e-8
    warmup_steps: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_interval: int = 2
    seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"
    wandb: bool = False


@dataclass
class PPOTrainConfig:
    """Minimal educational PPO-RLHF config."""
    model_name: str = "checkpoints/sft/hf"
    rm_out_dir: str = "checkpoints/rm"       # reward model checkpoint
    ppo_out_dir: str = "checkpoints/ppo"
    run_name: str = "ppo_qwen0.5b"
    data_dir: str = "data/dpo"
    max_seq_len: int = 256           # shorter for PPO generation speed
    max_new_tokens: int = 128
    # Reward model
    rm_epochs: int = 3
    rm_lr: float = 1.0e-5
    rm_batch_size: int = 2
    # PPO
    ppo_epochs: int = 2              # outer epochs over dataset
    ppo_steps_per_epoch: int = 20    # number of PPO update steps
    ppo_lr: float = 1.0e-6
    ppo_mini_batch: int = 4
    clip_eps: float = 0.2
    kl_coef: float = 0.1             # KL penalty vs SFT reference
    vf_coef: float = 0.5             # value function loss weight
    gamma: float = 1.0
    lam: float = 0.95                # GAE lambda
    seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"
