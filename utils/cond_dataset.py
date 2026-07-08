"""
Conditional dataset for the temporally-conditioned stream-function DDPM.

This is an ADDITIVE module — it does not modify the existing unconditional
pipeline (``dataset.OceanCurrentDataset``).  It supplies everything the
conditional stream-function model needs at training time:

    target field  x0                         (2, H, W)   ← the field to denoise
    conditioning  cond                       (C, H, W)   ← side information

The conditioning tensor stacks three groups, in this fixed order (see
``assemble_cond``):

    observation (soft):  obs_u, obs_v, path_mask, dist_to_path   4 channels
    temporal prior:      prev_L0(u,v), prev_L1(u,v), ...         2 * len(lags)
    geometry (static):   coord_x, coord_y, dist_coast[, bathy]   3 or 4 channels

Design notes
------------
* The dataset is backed by ONE continuous chronological field array (built from
  the raw ``.mat`` by ``utils/build_chrono_dataset.py``, format ``chrono_v1``).
  Each split is just a list of TARGET frame indices into that shared array, so a
  temporal prior is always the genuine earlier field ``fields[f - L]`` — never a
  cross-block fabrication.  The data is hourly, so a lag in frames equals a lag
  in hours.  Splits are separated by guard bands to prevent leakage.
* The observation channels reveal the TRUE field on a robot path.  At training
  time the path is a fresh random ``biased_walk_path`` (augmentation); at
  inference time the same channels are filled from the real measurements.
* dist_to_path gives each ocean cell its Euclidean distance to the nearest
  observed cell (normalized [0,1]).  This lets both models spatially localize
  uncertainty — cells far from the path should be treated differently.
* Prior fields are normalized with the SAME (mean, std) as the target — they are
  the same physical quantity — so the network sees a consistent scale.
* The geometry channels are derived for free from the land mask (no extra data).
  If a ``bathy`` array is stored in the pickle it is added as a 4th geometry
  channel (static, normalized to [0,1] over ocean cells).
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import OceanCurrentDataset   # reuse stats / normalization conventions
from paths   import biased_walk_path


# Data is hourly; 13 h and 25 h are same-tidal-phase peaks (see lag analysis).
DEFAULT_LAGS = (13, 25)

# Number of conditioning channels that are NOT the temporal priors.
_N_OBS_CH  = 4   # obs_u, obs_v, path_mask, dist_to_path
_N_GEOM_CH = 3   # coord_x, coord_y, dist_coast  (+ 1 if bathy present)


def cond_channels(lags=DEFAULT_LAGS, has_bathy: bool = False) -> int:
    """Total conditioning-channel count for a given set of lags."""
    geom_ch = _N_GEOM_CH + (1 if has_bathy else 0)
    return _N_OBS_CH + 2 * len(tuple(lags)) + geom_ch


def geometry_channels(land_mask: torch.Tensor,
                      bathy: np.ndarray | None = None) -> torch.Tensor:
    """
    Build static geometry channels from the land mask (and optional bathymetry).

    Args:
        land_mask: (H, W) bool tensor, True = land.
        bathy:     (H, W) float array of water depth (positive = deeper).
                   If provided, added as a 4th channel normalized to [0, 1]
                   over ocean cells; land cells are set to 0.

    Returns:
        (3 or 4, H, W) float32 tensor:
            coord_x    — normalized column position in [-1, 1]
            coord_y    — normalized row position in [-1, 1]
            dist_coast — distance from each ocean cell to nearest land,
                         normalized to [0, 1]; land cells are 0.
            bathy      — (optional) normalized water depth; land cells are 0.
    """
    from scipy import ndimage

    land = land_mask.cpu().numpy().astype(bool)
    H, W = land.shape
    ocean = ~land

    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)

    dist = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dmax = float(dist.max())
    if dmax > 0:
        dist = dist / dmax
    dist[land] = 0.0

    channels = [xs, ys, dist]

    if bathy is not None:
        b = np.asarray(bathy, dtype=np.float32).copy()
        b[land] = 0.0
        bmax = float(b[ocean].max()) if ocean.any() else 1.0
        if bmax > 0:
            b = b / bmax
        channels.append(b)

    geom = np.stack(channels, axis=0)               # (3 or 4, H, W)
    return torch.from_numpy(geom)


def observation_channels(field: torch.Tensor,
                         path_mask: np.ndarray,
                         land_np: np.ndarray) -> torch.Tensor:
    """
    Build the 4 soft-observation channels by revealing ``field`` on a path.

    Args:
        field:     (2, H, W) tensor — the (normalized) ground-truth field.
        path_mask: (H, W) bool array — True where the robot measured.
        land_np:   (H, W) bool array — True = land (for EDT masking).

    Returns:
        (4, H, W) float32 tensor: [obs_u, obs_v, path_mask, dist_to_path]
            obs_u / obs_v  — field components on the path, 0 elsewhere
            path_mask      — 1.0 on observed cells, 0.0 elsewhere
            dist_to_path   — Euclidean distance to nearest observed cell,
                             normalized to [0, 1] over ocean cells; land = 0.
    """
    from scipy import ndimage

    pm = torch.from_numpy(np.asarray(path_mask, dtype=bool))
    obs = torch.zeros_like(field)                    # (2, H, W)
    obs[:, pm] = field[:, pm]
    mask = pm.float()[None]                          # (1, H, W)

    # distance to nearest observed cell, normalized over ocean
    pm_np = pm.numpy()
    ocean_np = ~land_np
    dist = ndimage.distance_transform_edt(~pm_np).astype(np.float32)
    dist[land_np] = 0.0
    dmax = float(dist[ocean_np].max()) if ocean_np.any() and dist[ocean_np].max() > 0 else 1.0
    dist = dist / dmax
    dist[land_np] = 0.0
    dist_t = torch.from_numpy(dist)[None]            # (1, H, W)

    return torch.cat([obs, mask, dist_t], dim=0)     # (4, H, W)


def assemble_cond(obs: torch.Tensor,
                  priors: torch.Tensor,
                  geom: torch.Tensor) -> torch.Tensor:
    """
    Concatenate the conditioning groups in the canonical channel order.

    This is the SINGLE source of truth for channel ordering, used by both the
    dataset (training) and the inference code, so they can never disagree.

        [ obs (4) | priors (2*len(lags)) | geom (3 or 4) ]

    Args:
        obs:    (4, H, W)            observation channels
        priors: (2*len(lags), H, W)  temporal-prior channels
        geom:   (3 or 4, H, W)      static geometry channels

    Returns:
        (C, H, W) conditioning tensor.
    """
    return torch.cat([obs, priors, geom], dim=0)


class ConditionalOceanDataset(Dataset):
    """
    Time-paired ocean-current dataset for the conditional stream-function DDPM.

    Each item is a dict::

        {
          "target": (2, H, W),   # clean field x0 at frame f (to be denoised)
          "cond":   (C, H, W),   # assembled conditioning (see assemble_cond)
        }

    where C = 4 (obs) + 2*len(lags) (priors) + 3 or 4 (geom).

    Args:
        pickle_path: path to a ``chrono_v1`` pickle (built by
                     ``utils/build_chrono_dataset.py``): one continuous
                     chronological field array plus per-split target indices.
        split:       0/"train", 1/"val", 2/"test".
        lags:        prior lags in frames/hours (default (13, 25)).  Must be a
                     subset of (or no larger than) the lags the pickle was built
                     with, so every prior ``fields[f - L]`` is in range.
        data_mean:   normalization mean (applied to target AND priors).
        data_std:    normalization std.
        path_steps:  robot-path length.  An int for a fixed length, or a
                     (min, max) tuple to sample the length per item.
        deterministic: if True, the path RNG is seeded by the item index so
                     paths are reproducible (use for val/test).  If False, a
                     fresh random path is drawn each access (train augmentation).
        straight_bias: directional-persistence bias for ``biased_walk_path``.
    """

    _SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}

    def __init__(
        self,
        pickle_path: str,
        split=0,
        lags=DEFAULT_LAGS,
        data_mean:   float | None = None,
        data_std:    float | None = None,
        path_steps=150,
        deterministic: bool = False,
        straight_bias: float = 0.75,
    ):
        self.lags = tuple(int(l) for l in lags)
        if not self.lags:
            raise ValueError("lags must be non-empty")
        self.max_lag = max(self.lags)
        self.path_steps = path_steps
        self.deterministic = deterministic
        self.straight_bias = straight_bias

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        if not (isinstance(data, dict) and data.get("format") == "chrono_v1"):
            raise ValueError(
                f"{pickle_path} is not a chrono_v1 pickle; rebuild it with "
                "utils/build_chrono_dataset.py."
            )

        split_name = self._SPLIT_NAMES.get(split, split)
        if split_name not in data["splits"]:
            raise ValueError(
                f"split {split!r} -> {split_name!r} not in pickle "
                f"(have {list(data['splits'])})"
            )
        self.split = split
        self.split_name = split_name

        fields_np = np.asarray(data["fields"], dtype=np.float32)  # (N, 2, H, W)
        N = fields_np.shape[0]
        self.land_mask = torch.from_numpy(
            np.asarray(data["land_mask"], dtype=bool))
        self.fields = torch.from_numpy(np.nan_to_num(fields_np, nan=0.0))

        self.valid = np.asarray(data["splits"][split_name], dtype=np.int64)
        if self.valid.size == 0:
            raise ValueError(f"split {split_name!r} has no target frames")
        if int(self.valid.min()) < self.max_lag:
            raise ValueError(
                f"split {split_name!r} contains a target < max_lag={self.max_lag}; "
                f"requested lags {self.lags} exceed what the pickle was built "
                f"with ({tuple(data.get('lags', ()))})."
            )
        if int(self.valid.max()) >= N:
            raise ValueError(
                f"split {split_name!r} target index out of range for N={N}")

        self.data_mean = data_mean
        self.data_std  = data_std
        if data_mean is not None and data_std is not None:
            self.fields = (self.fields - data_mean) / max(float(data_std), 1e-8)
            self.fields[:, :, self.land_mask] = 0.0

        # Optional bathymetry stored in the pickle (added by build_chrono_dataset.py).
        bathy_np = data.get("bathy", None)
        if bathy_np is not None:
            bathy_np = np.asarray(bathy_np, dtype=np.float32)

        # Geometry is static — compute once.
        self.geom = geometry_channels(self.land_mask, bathy=bathy_np)  # (3 or 4, H, W)
        self.has_bathy = bathy_np is not None

        self._land_np = self.land_mask.cpu().numpy()

    @staticmethod
    def compute_stats(pickle_path: str, split: int = 0) -> tuple[float, float]:
        """Return (mean, std) used for normalization."""
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and data.get("format") == "chrono_v1":
            return float(data.get("data_mean", 0.0)), float(data["data_std"])
        return OceanCurrentDataset.compute_stats(pickle_path, split)

    def _sample_path_steps(self, rng: np.random.Generator) -> int:
        if isinstance(self.path_steps, (tuple, list)):
            lo, hi = int(self.path_steps[0]), int(self.path_steps[1])
            return int(rng.integers(lo, hi + 1))
        return int(self.path_steps)

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int) -> dict:
        f = int(self.valid[idx])
        target = self.fields[f]                              # (2, H, W)
        priors = torch.cat([self.fields[f - L] for L in self.lags], dim=0)

        seed = idx if self.deterministic else None
        rng  = np.random.default_rng(seed)
        n_steps  = self._sample_path_steps(rng)
        path_mask = biased_walk_path(
            self._land_np, n_steps=n_steps, seed=seed,
            straight_bias=self.straight_bias,
        )

        obs  = observation_channels(target, path_mask, self._land_np)  # (4, H, W)
        cond = assemble_cond(obs, priors, self.geom)                   # (C, H, W)
        return {"target": target, "cond": cond}
