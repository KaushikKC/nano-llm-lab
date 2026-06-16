from dataclasses import dataclass


@dataclass
class SFTConfig:
    """Hyperparameters for the Stage 2 supervised fine-tuning run."""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_seq_len: int = 512

    data_dir: str = "data/sft"
    out_dir: str = "checkpoints/sft"
    run_name: str = "sft_qwen0.5b"

    micro_batch_size: int = 2
    grad_accum_steps: int = 8
    epochs: int = 10

    lr: float = 2.0e-5
    min_lr: float = 2.0e-6
    warmup_steps: int = 10
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.999)

    log_interval: int = 5

    seed: int = 1337
    device: str = "auto"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False
    wandb: bool = False
