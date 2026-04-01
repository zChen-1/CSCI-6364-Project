from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_DIM = 124
TARGET_DIM = 128
LEVELS = 60
PROFILE_VARS = 2
SCALAR_VARS = 4
CHANNELS = PROFILE_VARS + SCALAR_VARS


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, array: np.ndarray) -> "Standardizer":
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean) / self.std

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return array * self.std + self.mean


class final_project_dataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset backed by NumPy arrays or memory-mapped .npy files."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_standardizer: Standardizer,
        y_standardizer: Standardizer,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        if x.shape[1] != INPUT_DIM:
            raise ValueError(f"Expected input dim {INPUT_DIM}, got {x.shape[1]}")
        if y.shape[1] != TARGET_DIM:
            raise ValueError(f"Expected target dim {TARGET_DIM}, got {y.shape[1]}")
        self.x = x
        self.y = y
        self.x_standardizer = x_standardizer
        self.y_standardizer = y_standardizer
        self.indices = np.arange(x.shape[0], dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row_idx = int(self.indices[idx])
        x_row = np.asarray(self.x[row_idx], dtype=np.float32)
        y_row = np.asarray(self.y[row_idx], dtype=np.float32)
        x_row = self.x_standardizer.transform(x_row)
        y_row = self.y_standardizer.transform(y_row)
        x_channels = convert_flat_input_to_channels(x_row)
        return torch.from_numpy(x_channels), torch.from_numpy(y_row)


def convert_flat_input_to_channels(x_row: np.ndarray) -> np.ndarray:
    """Convert a 124-dim quickstart input vector to a (6, 60) tensor.

    Channels are:
        0 temperature profile
        1 humidity profile
        2 repeated surface pressure
        3 repeated insolation
        4 repeated latent heat flux
        5 repeated sensible heat flux
    """
    if x_row.shape[0] != INPUT_DIM:
        raise ValueError(f"Expected flat input of length {INPUT_DIM}, got {x_row.shape[0]}")
    temp = x_row[:LEVELS]
    qv = x_row[LEVELS : 2 * LEVELS]
    scalars = x_row[2 * LEVELS :]
    repeated_scalars = np.repeat(scalars[:, None], LEVELS, axis=1)
    return np.concatenate([temp[None, :], qv[None, :], repeated_scalars], axis=0).astype(np.float32)


def load_npy_pair(data_dir: Path, x_name: str, y_name: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path = data_dir / x_name
    y_path = data_dir / y_name
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Could not find .npy files.\n")
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")

