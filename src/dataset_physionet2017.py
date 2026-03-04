"""
Phase 1: PhysioNet/CinC Challenge 2017 dataset loader.

Goal for Phase 1:
- Load records from disk reliably
- Clean and standardize shape for batching
- Return tensors usable by a 1D CNN

Expected directory layout (data_dir):
  data_dir/
    training/
      A00001.mat
      ...
    REFERENCE.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset

# Official labels for the 2017 challenge
LABEL_MAP = {"N": 0, "A": 1, "O": 2, "~": 3}


@dataclass
class PreprocessConfig:
    """
    Preprocessing settings.
    Keep these centralized so later we can add things like filtering,
    augmentation, resampling, etc.
    """
    target_len: int = 3000      # small = fast. Later you can use 9000 (30s @ 300Hz).
    do_zscore: bool = True      # per-record normalization


def _load_mat_signal(mat_path: Path) -> np.ndarray:
    """
    Load ECG signal from a .mat file.

    In this dataset, the signal is usually under 'val' with shape (1, n).
    """
    mat = scipy.io.loadmat(str(mat_path))
    if "val" not in mat:
        raise KeyError(f"'val' not found in {mat_path}. Keys: {list(mat.keys())}")
    x = mat["val"].squeeze()  # -> (n,)
    return x.astype(np.float32, copy=False)


def _clean_signal(x: np.ndarray) -> np.ndarray:
    """Replace NaNs and +/-Inf with 0.0 so training doesn't break."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _fix_length_center(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Make all samples the same length for batching.
    - If longer: center crop
    - If shorter: center pad with zeros
    """
    n = int(x.shape[0])

    if n == target_len:
        return x

    if n > target_len:
        start = (n - target_len) // 2
        return x[start : start + target_len]

    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(x, (pad_left, pad_right), mode="constant", constant_values=0.0).astype(np.float32)


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Standardize to mean 0, std 1. This is a simple, strong baseline.
    """
    mu = float(x.mean())
    sigma = float(x.std())
    return ((x - mu) / (sigma + eps)).astype(np.float32, copy=False)


class PhysioNet2017Dataset(Dataset):
    """
    Minimal dataset for Phase 1.

    Returns:
      x: FloatTensor with shape (1, target_len)  (channel-first for Conv1d)
      y: LongTensor scalar class index
    """

    def __init__(
        self,
        data_dir: str | Path,
        preprocess: PreprocessConfig,
        limit: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.preprocess = preprocess

        self.training_dir = self.data_dir / "training"
        self.label_csv = self.data_dir / "REFERENCE.csv"

        if not self.training_dir.exists():
            raise FileNotFoundError(f"Missing folder: {self.training_dir}")
        if not self.label_csv.exists():
            raise FileNotFoundError(f"Missing file: {self.label_csv}")

        # REFERENCE.csv is typically: record_id,label (no header)
        df = pd.read_csv(self.label_csv, header=None, names=["record", "label"])
        df["record"] = df["record"].astype(str)
        df["label"] = df["label"].astype(str)

        # Keep only official label set
        df = df[df["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)

        # Optional: limit dataset size for a fast Phase 1 smoke test
        if limit is not None and limit > 0:
            df = df.iloc[: int(limit)].reset_index(drop=True)

        self.records: List[str] = df["record"].tolist()
        self.labels: List[int] = [LABEL_MAP[lbl] for lbl in df["label"].tolist()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record_id = self.records[idx]
        label_id = self.labels[idx]

        mat_path = self.training_dir / f"{record_id}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing record file: {mat_path}")

        x = _load_mat_signal(mat_path)
        x = _clean_signal(x)
        x = _fix_length_center(x, self.preprocess.target_len)

        if self.preprocess.do_zscore:
            x = _zscore(x)

        # Convert to torch tensor, add channel dimension for Conv1d
        x_t = torch.from_numpy(x).unsqueeze(0)               # (1, L)
        y_t = torch.tensor(label_id, dtype=torch.long)       # ()

        return x_t, y_t
