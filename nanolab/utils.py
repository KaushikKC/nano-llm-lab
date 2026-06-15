import random

import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    """Resolve "auto" to the best available accelerator: MPS (Apple Silicon),
    CUDA (NVIDIA), or CPU, in that order."""
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(module: torch.nn.Module, exclude: set | None = None) -> int:
    """Count trainable parameters, optionally excluding a set of tensors
    (e.g. a tied embedding/lm_head weight, which would otherwise be counted
    once per reference to the same underlying storage — but `parameters()`
    already de-duplicates by identity, so `exclude` is for reporting
    "non-embedding" counts deliberately)."""
    exclude = exclude or set()
    seen = set()
    total = 0
    for p in module.parameters():
        if id(p) in seen or id(p) in {id(e) for e in exclude}:
            continue
        seen.add(id(p))
        total += p.numel()
    return total
