from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO (Group Relative Policy Optimisation) Rung 3.

    Requires CUDA — MPS doesn't support the online rollout memory budget.
    Reference: DeepSeek-R1 / GRPO paper (Shao et al. 2024).
    """
    model_name: str = "checkpoints/tool_dpo/hf"
    ref_model_name: str = "checkpoints/tool_dpo/hf"
    data_dir: str = "data/tool_dpo"
    out_dir: str = "checkpoints/tool_grpo"
    run_name: str = "tool_grpo_qwen0.5b"

    # GRPO rollout
    group_size: int = 4        # G responses sampled per prompt
    temperature: float = 0.8   # sampling temperature for rollouts

    # Ternary reward: +1 correct tool, 0 abstain, -1 hallucination
    reward_correct: float = 1.0
    reward_abstain: float = 0.0
    reward_hallucination: float = -1.0

    # Training
    max_seq_len: int = 384
    micro_batch_size: int = 1
    grad_accum_steps: int = 4
    epochs: int = 3
    lr: float = 1.0e-6
    min_lr: float = 1.0e-7
    warmup_steps: int = 5
    kl_coef: float = 0.04      # KL penalty against reference
    clip_eps: float = 0.2
    grad_clip: float = 1.0
    log_interval: int = 2
    seed: int = 42
    device: str = "cuda"       # CUDA required
    dtype: str = "bfloat16"
    wandb: bool = False
