from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraTrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    # LoRA adapter
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    # Data
    max_seq_len: int = 512
    data_dir: str = "data/sft"
    # Training
    out_dir: str = "checkpoints/lora"
    run_name: str = "lora_qwen0.5b"
    micro_batch_size: int = 2
    grad_accum_steps: int = 8
    epochs: int = 10
    lr: float = 2.0e-4   # LoRA typically uses higher LR than full FT
    min_lr: float = 2.0e-5
    warmup_steps: int = 10
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_interval: int = 5
    seed: int = 1337
    device: str = "auto"
    dtype: str = "bfloat16"
    wandb: bool = False


@dataclass
class QLoraTrainConfig(LoraTrainConfig):
    # 4-bit quantization settings
    bnb_4bit_quant_type: str = "nf4"          # NormalFloat4 — better for LLM weights
    bnb_4bit_use_double_quant: bool = True     # quantize the quantization constants
    bnb_4bit_compute_dtype: str = "float16"   # compute in fp16 during forward pass
    out_dir: str = "checkpoints/qlora"
    run_name: str = "qlora_qwen0.5b"
