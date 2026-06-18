"""Unit tests for DPO dataset and config — no model downloads required."""
import pytest
import torch

from nanolab.dpo.config import DPOTrainConfig, PPOTrainConfig
from nanolab.dpo.dataset import dpo_collate


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_dpo_config_defaults():
    cfg = DPOTrainConfig()
    assert cfg.beta == 0.1
    assert cfg.epochs == 3
    assert cfg.lr == 5.0e-7
    assert cfg.micro_batch_size == 1


def test_ppo_config_defaults():
    cfg = PPOTrainConfig()
    assert cfg.clip_eps == 0.2
    assert cfg.kl_coef == 0.1
    assert cfg.ppo_epochs == 2


# ---------------------------------------------------------------------------
# dpo_collate tests (no tokenizer needed — works on raw int lists)
# ---------------------------------------------------------------------------

def _make_example(ids_w, ids_l):
    return {
        "input_ids_w": ids_w,
        "labels_w":    [-100] * 2 + ids_w[2:],  # first 2 tokens masked as prompt
        "input_ids_l": ids_l,
        "labels_l":    [-100] * 2 + ids_l[2:],
    }


def test_collate_single_example():
    ex = _make_example([1, 2, 3, 4], [1, 2, 5, 6])
    batch = dpo_collate([ex], pad_id=0)
    assert batch["input_ids_w"].shape == (1, 4)
    assert batch["labels_w"][0, 0].item() == -100
    assert batch["labels_w"][0, 2].item() == 3


def test_collate_pads_to_max_length():
    ex1 = _make_example([1, 2, 3], [1, 2, 4])
    ex2 = _make_example([1, 2, 3, 4, 5], [1, 2, 6, 7, 8])
    batch = dpo_collate([ex1, ex2], pad_id=0)
    # Shorter sequence should be padded
    assert batch["input_ids_w"].shape == (2, 5)
    assert batch["input_ids_w"][0, 3].item() == 0  # padded
    assert batch["input_ids_w"][0, 4].item() == 0  # padded
    assert batch["labels_w"][0, 3].item() == -100  # padded labels = -100


def test_collate_labels_padding_is_minus100():
    ex1 = _make_example([1, 2, 3], [1, 2, 4])
    ex2 = _make_example([1, 2, 3, 4, 5], [1, 2, 6, 7, 8])
    batch = dpo_collate([ex1, ex2], pad_id=99)
    # Padding positions in labels must be -100 (not 99)
    assert batch["labels_w"][0, 3].item() == -100
    assert batch["labels_l"][0, 3].item() == -100


def test_collate_output_types():
    ex = _make_example([10, 20], [10, 30])
    batch = dpo_collate([ex], pad_id=0)
    for key in ("input_ids_w", "labels_w", "input_ids_l", "labels_l"):
        assert isinstance(batch[key], torch.Tensor)
        assert batch[key].dtype == torch.long
