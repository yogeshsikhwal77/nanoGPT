import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from tokenizers import Tokenizer


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



class SFTDataset(Dataset):
    def __init__(self, data_path: str, context_len: int, tokenizer_path: str = "tokenizer.json"):
        self.context_len = context_len
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        story = item.get("story", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        prompt_text = f"<|story|> {story} <|question|> {question} <|answer|>"
        answer_text = f" {answer}"
        
        prompt_ids = self.tokenizer.encode(prompt_text).ids
        answer_ids = self.tokenizer.encode(answer_text).ids + [self.eos_id]
        
        total_len = len(prompt_ids) + len(answer_ids)
        if total_len > self.context_len + 1:
            excess = total_len - (self.context_len + 1)
            prompt_ids = prompt_ids[excess:]
            
        x_ids = prompt_ids + answer_ids
        y_ids = ([-100] * len(prompt_ids)) + answer_ids
        
        # Pad to context length using pad_id
        pad_len = (self.context_len + 1) - len(x_ids)
        if pad_len > 0:
            x_ids.extend([self.pad_id] * pad_len)
            y_ids.extend([-100] * pad_len)
            
        x = torch.tensor(x_ids[:-1], dtype=torch.long)
        y = torch.tensor(y_ids[1:], dtype=torch.long)
        
        return x, y