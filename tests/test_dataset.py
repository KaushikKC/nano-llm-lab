import numpy as np
import pytest
import torch

from nanolab.data.dataset import TokenDataset


@pytest.fixture
def bin_path(tmp_path):
    # A simple increasing sequence so we can verify x/y alignment exactly.
    tokens = np.arange(1000, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    tokens.tofile(path)
    return str(path)


def test_too_short_dataset_raises(tmp_path):
    tokens = np.arange(10, dtype=np.uint16)
    path = tmp_path / "short.bin"
    tokens.tofile(path)
    with pytest.raises(ValueError):
        TokenDataset(str(path), block_size=20)


def test_get_batch_shapes(bin_path):
    ds = TokenDataset(bin_path, block_size=16)
    x, y = ds.get_batch(batch_size=8, device=torch.device("cpu"))
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_targets_are_inputs_shifted_by_one(bin_path):
    ds = TokenDataset(bin_path, block_size=16)
    x, y = ds.get_batch(batch_size=4, device=torch.device("cpu"))
    # Our fixture data is `arange`, so y should equal x + 1 everywhere.
    assert torch.equal(y, x + 1)


def test_indices_stay_in_bounds(bin_path):
    ds = TokenDataset(bin_path, block_size=16)
    for _ in range(20):
        x, y = ds.get_batch(batch_size=4, device=torch.device("cpu"))
        assert x.max().item() <= 999
        assert y.max().item() <= 999
