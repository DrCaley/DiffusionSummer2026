from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from model_parameters.noise_types import project_divergence_free_sample


def load_pickle_splits(pickle_path: str | Path) -> list[np.ndarray]:
    with open(pickle_path, "rb") as handle:
        splits = pickle.load(handle)
    if not isinstance(splits, list) or len(splits) != 3:
        raise ValueError("Expected a pickle list with train/val/test arrays")
    return splits


def save_pickle_splits(splits: list[np.ndarray], pickle_path: str | Path) -> None:
    with open(pickle_path, "wb") as handle:
        pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_land_mask(split: np.ndarray) -> np.ndarray:
    if split.ndim != 4 or split.shape[2] != 2:
        raise ValueError(f"Expected split shape (H, W, 2, N), got {split.shape}")
    return np.isnan(split).any(axis=(2, 3))


def _to_nchw(split: np.ndarray) -> np.ndarray:
    return np.nan_to_num(split, nan=0.0).transpose(3, 2, 0, 1).astype(np.float32, copy=False)


def prepare_divergence_free_pickle(source_pickle: str | Path, target_pickle: str | Path) -> np.ndarray:
    source_path = Path(source_pickle)
    target_path = Path(target_pickle)
    splits = load_pickle_splits(source_path)
    land_mask = build_land_mask(splits[0])

    cleaned_splits: list[np.ndarray] = []
    for split in splits:
        nchw = _to_nchw(split)
        projected_samples = []
        for sample in nchw:
            projected = project_divergence_free_sample(sample)
            projected[:, land_mask] = np.nan
            projected_samples.append(projected)
        projected_nchw = np.stack(projected_samples, axis=0)
        cleaned_splits.append(projected_nchw.transpose(2, 3, 1, 0).astype(np.float32, copy=False))

    save_pickle_splits(cleaned_splits, target_path)
    return land_mask


def load_cleaned_pickle(pickle_path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
    splits = load_pickle_splits(pickle_path)
    land_mask = build_land_mask(splits[0])
    return splits, land_mask


class OceanCurrentDataset(Dataset):
    def __init__(self, split: np.ndarray, land_mask: np.ndarray):
        self.samples = np.nan_to_num(split, nan=0.0).transpose(3, 2, 0, 1).astype(np.float32, copy=False)
        self.land_mask = torch.from_numpy(land_mask.astype(bool, copy=False))

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = torch.from_numpy(self.samples[index])
        return {"field": sample, "land_mask": self.land_mask}


def field_to_numpy(field: torch.Tensor) -> np.ndarray:
    if field.dim() != 3 or field.shape[0] != 2:
        raise ValueError(f"Expected tensor shape (2, H, W), got {tuple(field.shape)}")
    return field.detach().cpu().numpy()


def split_metadata(splits: list[np.ndarray]) -> dict[str, Any]:
    return {
        "train_samples": int(splits[0].shape[-1]),
        "val_samples": int(splits[1].shape[-1]),
        "test_samples": int(splits[2].shape[-1]),
        "height": int(splits[0].shape[0]),
        "width": int(splits[0].shape[1]),
    }
