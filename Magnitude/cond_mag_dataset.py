"""
Conditioned magnitude (speed) dataset — Phase 2.

Gives the deterministic speed regressor the SAME 10-channel conditioning stack
the conditional diffusion model consumes (obs + temporal priors + geometry),
instead of the original 3-channel [obs_speed, path_mask, land] input.

Motivation: the 3-channel UNet has no signal for a frame's overall energy
level, so it over-predicts speed in unobserved regions on weak-flow days.  The
temporal-prior channels (prev 13 h / 25 h fields) carry exactly that energy
signal, fixing the far-field overshoot.

Each item is ``(cond, target_speed, ocean)``:
    cond          (C, H, W)  conditioning, C = cond_channels(lags) (=10 default),
                             normalized by data_std (identical to the diffusion
                             model's input).
    target_speed  (1, H, W)  dense speed |v|, standardized by (speed_mean,
                             speed_std) in PHYSICAL units — the SAME output
                             convention as the original Magnitude_UNet so the
                             fusion code is unchanged.
    ocean         (1, H, W)  1.0 on ocean cells, 0.0 on land.

The path baked into ``cond`` is random per access (train augmentation) or
index-seeded (deterministic val), inherited from ``ConditionalOceanDataset``.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_root, os.path.join(_root, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cond_dataset import ConditionalOceanDataset, cond_channels  # noqa: E402


def speed_stats_chrono(pickle_path: str, data_std: float,
                       lags=(13, 25), split: int = 0) -> tuple[float, float]:
    """
    Mean and std of PHYSICAL ocean-cell speed |v| over a chrono split's target
    frames.  ``ds.fields`` are data_std-normalized, so physical = field*data_std.
    """
    ds = ConditionalOceanDataset(pickle_path, split=split, lags=lags,
                                 data_mean=0.0, data_std=data_std)
    ocean = (~ds.land_mask.cpu().numpy()).astype(bool)
    fields = ds.fields.cpu().numpy()
    acc = sq = 0.0
    n = 0
    for f in ds.valid:
        fld = fields[int(f)] * data_std                       # physical units
        sp = np.sqrt(fld[0] ** 2 + fld[1] ** 2)[ocean]
        acc += float(sp.sum()); sq += float((sp ** 2).sum()); n += sp.size
    mean = acc / max(n, 1)
    var = max(sq / max(n, 1) - mean ** 2, 1e-12)
    return mean, float(np.sqrt(var))


class CondMagnitudeDataset(Dataset):
    """Conditioning -> dense speed pairs for the conditioned magnitude UNet."""

    def __init__(self, pickle_path: str, split: int, speed_mean: float,
                 speed_std: float, data_std: float, lags=(13, 25),
                 path_steps=(30, 200), deterministic: bool = False):
        self.ds = ConditionalOceanDataset(
            pickle_path, split=split, lags=lags, data_mean=0.0,
            data_std=data_std, path_steps=path_steps,
            deterministic=deterministic)
        self.land = self.ds.land_mask.cpu().numpy().astype(bool)
        self.ocean = ~self.land
        self.speed_mean = float(speed_mean)
        self.speed_std = float(speed_std)
        self.data_std = float(data_std)
        self.cond_ch = cond_channels(lags)
        self._ocean_ch = torch.from_numpy(self.ocean.astype(np.float32))[None]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        cond = item["cond"]                                   # (C, H, W)
        tgt = item["target"].cpu().numpy()                    # (2, H, W), data_std units
        spd_phys = np.sqrt(tgt[0] ** 2 + tgt[1] ** 2) * self.data_std
        target = (spd_phys - self.speed_mean) / self.speed_std
        target[self.land] = 0.0
        return (cond,
                torch.from_numpy(target.astype(np.float32))[None],
                self._ocean_ch)
