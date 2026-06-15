import numpy as np
import torch


class TokenDataset:
    """A flat array of token ids on disk, memmapped for cheap random access.

    Rather than a `torch.utils.data.Dataset` + `DataLoader`, training simply
    samples random starting offsets and slices out fixed-length windows —
    the standard nanoGPT approach. This avoids any indexing/shuffling
    overhead and works well for a single contiguous token stream.
    """

    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) <= block_size:
            raise ValueError(
                f"dataset {bin_path} has only {len(self.data)} tokens, "
                f"need more than block_size={block_size}"
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample `batch_size` random (x, y) windows, where y is x shifted
        right by one token (the next-token-prediction target)."""
        max_start = len(self.data) - self.block_size - 1
        starts = np.random.randint(0, max_start, size=batch_size)

        x = np.stack([self.data[i : i + self.block_size] for i in starts])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size] for i in starts])

        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        if device.type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y
