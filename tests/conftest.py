import pytest
import torch

from nanolab.config import GPTConfig


@pytest.fixture
def tiny_config() -> GPTConfig:
    """A deliberately tiny config so model tests run in milliseconds on CPU."""
    return GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=2,
        d_model=32,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )


@pytest.fixture(autouse=True)
def _force_cpu_and_seed():
    torch.manual_seed(0)
    yield
