from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Architecture / scaling hyperparameters for the GPT model."""

    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1024
    max_seq_len: int = 256
    rope_base: float = 10000.0
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Optimization / data / logging hyperparameters for the training run."""

    data_dir: str = "data/processed"
    out_dir: str = "checkpoints/small"
    run_name: str = "small"

    micro_batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 20000

    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 300
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)

    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    ckpt_interval: int = 1000

    seed: int = 1337
    device: str = "auto"
    amp: bool = False
    compile: bool = False
    wandb: bool = False
