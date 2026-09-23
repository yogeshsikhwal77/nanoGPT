import os
import glob
import numpy as np
import torch


class MemmapPretrainDataset:
    """
    Reads shard_*.bin files (raw uint16 token ids, written by tokenizers.encode)
    and samples fixed-length (context_len + 1) windows for next-token prediction.
    """

    def __init__(self, data_dir: str, context_len: int, split: str = "train", val_fraction: float = 0.1):
        assert split in ("train", "val")
        self.context_len = context_len
        self.split = split
        self.val_fraction = val_fraction

        paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.bin")))
        if not paths:
            raise ValueError(f"No shard_*.bin files found in {data_dir}")
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]

    def _usable_ranges(self):
        # last val_fraction of each shard is held out for validation
        ranges = []
        for s in self.shards:
            n = len(s)
            split_point = int(n * (1 - self.val_fraction))
            ranges.append((0, split_point) if self.split == "train" else (split_point, n))
        return ranges

    def get_batch(self, batch_size: int, device: str = "cpu"):
        """Returns (x, y), each shaped (batch_size, context_len). y is x shifted by one token."""
        ranges = self._usable_ranges()
        weights = np.array([hi - lo for lo, hi in ranges], dtype=np.float64)
        weights /= weights.sum()

        xs, ys = [], []
        for _ in range(batch_size):
            shard_idx = np.random.choice(len(self.shards), p=weights)
            shard = self.shards[shard_idx]
            lo, hi = ranges[shard_idx]

            max_start = hi - self.context_len - 1
            if max_start <= lo:
                raise ValueError(
                    f"Shard {shard_idx} too short for context_len={self.context_len} "
                    f"in split='{self.split}' ({hi - lo} usable tokens)"
                )
            start = np.random.randint(lo, max_start)
            chunk = shard[start : start + self.context_len + 1].astype(np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])

        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        return x.to(device), y.to(device)


class TinyOverfitDataset:
    """
    Freezes a fixed pool of `num_samples` windows from the train split.
    Used only for the tiny-overfit sanity check — loss on this fixed pool
    should collapse toward 0 within a few hundred steps.
    """

    def __init__(self, data_dir: str, context_len: int, num_samples: int = 100, seed: int = 42):
        np.random.seed(seed)
        base = MemmapPretrainDataset(data_dir, context_len, split="train")
        self.x, self.y = base.get_batch(num_samples)

    def get_batch(self, batch_size: int, device: str = "cpu"):
        idx = np.random.randint(0, self.x.shape[0], size=batch_size)
        return self.x[idx].to(device), self.y[idx].to(device)