"""
dataset.py – Ocean current data loading for the Colored-Noise DDPM.

Data layout (data.pickle):
  List of 3 arrays, one per split (train / val / test).
  Each array shape: (X=94, Y=44, C=2, N)
    X  – east-west grid dimension
    Y  – north-south grid dimension
    C  – velocity channel: 0=u (east-west), 1=v (north-south)
    N  – number of time snapshots

After transposing for PyTorch: (N, 2, H=94, W=44)

Land pixels contain NaN – replaced with 0.0 at load time.
The land mask (True = land) is constant across all snapshots and splits.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(pickle_path: str):
    """Load and preprocess the ocean current pickle file.

    Returns
    -------
    splits    : list of 3 float32 ndarrays, each (N, 2, 94, 44)
    land_mask : BoolTensor (94, 44)  – True = land pixel
    """
    with open(pickle_path, "rb") as fh:
        raw = pickle.load(fh)

    splits = []
    for arr in raw:
        # (94, 44, 2, N) → (N, 2, 94, 44)
        t = arr.transpose(3, 2, 0, 1).astype(np.float32)
        splits.append(t)

    # Land mask: NaN pattern is identical across all splits and timesteps
    land_mask_np = np.isnan(splits[0][0, 0])   # (94, 44) bool

    # Replace NaN → 0
    splits = [np.nan_to_num(s, nan=0.0) for s in splits]

    land_mask = torch.from_numpy(land_mask_np)
    return splits, land_mask


class OceanCurrentDataset(Dataset):
    """Wraps a (N, 2, H, W) float32 array as a PyTorch Dataset."""

    def __init__(self, data_array: np.ndarray):
        self.data = torch.from_numpy(data_array)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def get_dataloaders(pickle_path: str, batch_size: int = 32, num_workers: int = 0):
    """Build train / val / test DataLoaders and return the land mask.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    land_mask : BoolTensor (H, W)  – True = land
    """
    splits, land_mask = load_data(pickle_path)

    train_ds = OceanCurrentDataset(splits[0])
    val_ds   = OceanCurrentDataset(splits[1])
    test_ds  = OceanCurrentDataset(splits[2])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, land_mask
