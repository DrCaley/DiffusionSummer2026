"""
Dataset wrapper that turns ocean-current snapshots into sparse→dense speed
inpainting pairs for the magnitude UNet.

For each snapshot we:
  1. compute the ground-truth speed field |v| = sqrt(u^2 + v^2),
  2. sample a *fresh* biased random-walk robot path (different every epoch),
  3. build the 3-channel input [observed_speed, path_mask, land_mask],
  4. return (input, target_speed, ocean_mask) with speed standardized.

Sampling a new path on every __getitem__ acts as strong data augmentation:
the network sees each field revealed through many different trajectories and
must learn the general sparse→dense mapping rather than memorizing one path.

Standardization: the observed-speed input channel and the target are both
divided by `speed_std` (after subtracting `speed_mean` from the target only —
the input keeps a true zero at unobserved cells, so we scale it without
shifting).  The training script computes (speed_mean, speed_std) once over the
training split's ocean cells and stores them in the checkpoint.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "utils"))

from dataset import OceanCurrentDataset  # noqa: E402
from paths   import biased_walk_path      # noqa: E402


def speed_stats(pickle_path: str, split: int = 0) -> tuple[float, float]:
    """Mean and std of ocean-cell speed |v| over a dataset split."""
    base = OceanCurrentDataset(pickle_path, split=split)
    land = base.land_mask.numpy()
    ocean = ~land
    acc, sq, n = 0.0, 0.0, 0
    for i in range(len(base)):
        x0 = base[i].numpy()
        sp = np.sqrt(x0[0] ** 2 + x0[1] ** 2)[ocean]
        acc += float(sp.sum())
        sq  += float((sp ** 2).sum())
        n   += sp.size
    mean = acc / n
    var  = max(sq / n - mean ** 2, 1e-12)
    return mean, float(np.sqrt(var))


class MagnitudeDataset(Dataset):
    """Sparse→dense speed inpainting pairs with on-the-fly random paths."""

    def __init__(
        self,
        pickle_path: str,
        split:       int,
        speed_mean:  float,
        speed_std:   float,
        path_steps:  int  = 150,
        fixed_paths: bool = False,
        seed:        int  = 0,
    ):
        """
        Args:
            pickle_path: path to data.pickle
            split:       0 = train, 1 = val, 2 = test
            speed_mean:  training-split mean ocean speed (for standardizing target)
            speed_std:   training-split std  ocean speed (for standardizing both)
            path_steps:  robot walk length
            fixed_paths: if True, use a deterministic per-index path (for stable
                         validation); if False, sample a fresh random path each
                         call (training augmentation)
            seed:        base seed for fixed_paths mode
        """
        self.base        = OceanCurrentDataset(pickle_path, split=split)
        self.land_mask   = self.base.land_mask.numpy().astype(bool)   # (H, W)
        self.ocean_mask  = ~self.land_mask
        self.speed_mean  = float(speed_mean)
        self.speed_std   = float(speed_std)
        self.path_steps  = path_steps
        self.fixed_paths = fixed_paths
        self.seed        = seed

        self.land_ch = torch.from_numpy(self.land_mask.astype(np.float32))[None]  # (1,H,W)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x0  = self.base[idx].numpy()                       # (2, H, W)
        spd = np.sqrt(x0[0] ** 2 + x0[1] ** 2).astype(np.float32)  # (H, W)
        spd[self.land_mask] = 0.0

        # Sample a robot path. Fixed (deterministic) for val, random for train.
        path_seed = (self.seed + idx) if self.fixed_paths else None
        path_mask = biased_walk_path(self.land_mask, n_steps=self.path_steps, seed=path_seed)
        path_mask &= self.ocean_mask

        # ---- Input channels (standardized speed by std only; true zero kept) ----
        obs_speed = np.zeros_like(spd)
        obs_speed[path_mask] = spd[path_mask] / self.speed_std

        inp = np.stack([
            obs_speed,                              # observed speed (scaled)
            path_mask.astype(np.float32),           # path mask
            self.land_mask.astype(np.float32),      # land mask
        ], axis=0)                                  # (3, H, W)

        # ---- Target (standardized: subtract mean, divide by std) ----
        target = (spd - self.speed_mean) / self.speed_std
        target[self.land_mask] = 0.0
        target = target[None]                       # (1, H, W)

        ocean = self.ocean_mask.astype(np.float32)[None]  # (1, H, W)

        return (
            torch.from_numpy(inp),
            torch.from_numpy(target),
            torch.from_numpy(ocean),
        )
