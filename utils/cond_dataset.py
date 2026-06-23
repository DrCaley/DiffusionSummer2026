"""
Conditional dataset for the temporally-conditioned stream-function DDPM.

This is an ADDITIVE module — it does not modify the existing unconditional
pipeline (``dataset.OceanCurrentDataset``).  It supplies everything the
conditional stream-function model needs at training time:

    target field  x0                         (2, H, W)   ← the field to denoise
    conditioning  cond                       (C, H, W)   ← side information

The conditioning tensor stacks three groups, in this fixed order (see
``assemble_cond``):

    observation (soft):  obs_u, obs_v, path_mask            3 channels
    temporal prior:      prev_L0(u,v), prev_L1(u,v), ...    2 * len(lags)
    geometry (static):   coord_x, coord_y, dist_coast       3 channels

Design notes
------------
* Splits are INDEPENDENT contiguous time segments (verified: train/val/test do
  not form one continuous timeline), so temporal pairs are formed WITHIN a split
  only.  The first ``max(lags)`` frames of a split have no in-split prior and are
  dropped.  The data is hourly, so a lag in frames equals a lag in hours.
* The observation channels reveal the TRUE field on a robot path.  At training
  time the path is a fresh random ``biased_walk_path`` (augmentation); at
  inference time the same channels are filled from the real measurements.
* Prior fields are normalized with the SAME (mean, std) as the target — they are
  the same physical quantity — so the network sees a consistent scale.
* The geometry channels are derived for free from the land mask (no extra data).
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
_N_OBS_CH  = 3   # obs_u, obs_v, path_mask
_N_GEOM_CH = 3   # coord_x, coord_y, dist_coast


def cond_channels(lags=DEFAULT_LAGS) -> int:
    """Total conditioning-channel count for a given set of lags."""
    return _N_OBS_CH + 2 * len(tuple(lags)) + _N_GEOM_CH


def geometry_channels(land_mask: torch.Tensor) -> torch.Tensor:
    """
    Build the 3 static geometry channels from the land mask.

    Args:
        land_mask: (H, W) bool tensor, True = land.

    Returns:
        (3, H, W) float32 tensor: [coord_x, coord_y, dist_coast]
            coord_x    — normalized column position in [-1, 1]
            coord_y    — normalized row position in [-1, 1]
            dist_coast — distance from each ocean cell to nearest land,
                         normalized to [0, 1]; land cells are 0.
    """
    from scipy import ndimage

    land = land_mask.cpu().numpy().astype(bool)
    H, W = land.shape
    ocean = ~land

    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)

    # EDT of the ocean mask: distance from each ocean cell to the nearest land
    # (zero) cell.  Land cells are 0 in the input, hence 0 distance.
    dist = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dmax = float(dist.max())
    if dmax > 0:
        dist = dist / dmax
    dist[land] = 0.0

    geom = np.stack([xs, ys, dist], axis=0)            # (3, H, W)
    return torch.from_numpy(geom)


def observation_channels(field: torch.Tensor,
                         path_mask: np.ndarray) -> torch.Tensor:
    """
    Build the 3 soft-observation channels by revealing ``field`` on a path.

    Args:
        field:     (2, H, W) tensor — the (normalized) ground-truth field.
        path_mask: (H, W) bool array — True where the robot measured.

    Returns:
        (3, H, W) float32 tensor: [obs_u, obs_v, path_mask]
            obs_u / obs_v — field components on the path, 0 elsewhere
            path_mask     — 1.0 on observed cells, 0.0 elsewhere
    """
    pm = torch.from_numpy(np.asarray(path_mask, dtype=bool))
    obs = torch.zeros_like(field)                       # (2, H, W)
    obs[:, pm] = field[:, pm]
    mask = pm.float()[None]                             # (1, H, W)
    return torch.cat([obs, mask], dim=0)                # (3, H, W)


def assemble_cond(obs: torch.Tensor,
                  priors: torch.Tensor,
                  geom: torch.Tensor) -> torch.Tensor:
    """
    Concatenate the conditioning groups in the canonical channel order.

    This is the SINGLE source of truth for channel ordering, used by both the
    dataset (training) and the inference code, so they can never disagree.

        [ obs (3) | priors (2*len(lags)) | geom (3) ]

    Args:
        obs:    (3, H, W)            observation channels
        priors: (2*len(lags), H, W)  temporal-prior channels
        geom:   (3, H, W)            static geometry channels

    Returns:
        (C, H, W) conditioning tensor.
    """
    return torch.cat([obs, priors, geom], dim=0)


# Adjacent-frame correlation below this marks a discontinuity (block boundary).
# Within a continuous block neighbours correlate ~0.85-0.96; at a boundary the
# correlation collapses to near 0 or negative (verified empirically).
DEFAULT_BLOCK_THRESHOLD = 0.5


def detect_block_ids(fields: np.ndarray,
                     ocean_mask: np.ndarray,
                     threshold: float = DEFAULT_BLOCK_THRESHOLD) -> np.ndarray:
    """
    Assign each frame to a contiguous-in-time block.

    The pickle splits are NOT single continuous timelines: each split is a
    concatenation of short contiguous blocks pulled from across the whole
    record.  A temporal prior (frame ``f - L``) is only physically valid when
    it lies in the SAME block as the target ``f``.  This function labels frames
    so callers can enforce that.

    Detection is by adjacent-frame Pearson correlation over ocean cells: a drop
    below ``threshold`` between frame ``i`` and ``i+1`` starts a new block.

    Args:
        fields:     (N, 2, H, W) array of (possibly normalized) fields.
        ocean_mask: (H, W) bool array, True = ocean.
        threshold:  correlation below which a boundary is declared.

    Returns:
        (N,) int64 array of block ids (0, 1, 2, ...); equal ids = same block.
    """
    fields = np.asarray(fields)
    N = fields.shape[0]
    if N <= 1:
        return np.zeros(N, dtype=np.int64)
    ocean = np.asarray(ocean_mask, dtype=bool)
    flat = fields[:, :, ocean].reshape(N, -1).astype(np.float64)  # (N, 2*ocean)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat /= (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    cons = np.einsum("ij,ij->i", flat[:-1], flat[1:])            # (N-1,) adj corr
    is_break = cons < threshold
    block = np.zeros(N, dtype=np.int64)
    block[1:] = np.cumsum(is_break)
    return block


def valid_block_targets(block_ids: np.ndarray, lags) -> np.ndarray:
    """
    Indices of target frames whose prior at every lag is in the same block.

    Args:
        block_ids: (N,) block id per frame (from ``detect_block_ids``).
        lags:      iterable of integer lags.

    Returns:
        int64 array of valid target frame indices.
    """
    lags = tuple(int(l) for l in lags)
    max_lag = max(lags)
    N = len(block_ids)
    f = np.arange(max_lag, N, dtype=np.int64)
    keep = np.ones_like(f, dtype=bool)
    for L in lags:
        keep &= block_ids[f - L] == block_ids[f]
    return f[keep]


class ConditionalOceanDataset(Dataset):
    """
    Time-paired ocean-current dataset for the conditional stream-function DDPM.

    Each item is a dict::

        {
          "target": (2, H, W),   # clean field x0 at frame f (to be denoised)
          "cond":   (C, H, W),   # assembled conditioning (see assemble_cond)
        }

    where C = 3 (obs) + 2*len(lags) (priors) + 3 (geom).

    Args:
        pickle_path: path to the data pickle (list of 3 split arrays).
        split:       0 = train, 1 = val, 2 = test.
        lags:        prior lags in frames/hours (default (13, 25)).
        data_mean:   normalization mean (applied to target AND priors).
        data_std:    normalization std.
        path_steps:  robot-path length.  An int for a fixed length, or a
                     (min, max) tuple to sample the length per item.
        deterministic: if True, the path RNG is seeded by the item index so
                     paths are reproducible (use for val/test).  If False, a
                     fresh random path is drawn each access (train augmentation).
        straight_bias: directional-persistence bias for ``biased_walk_path``.
    """

    def __init__(
        self,
        pickle_path: str,
        split:       int = 0,
        lags=DEFAULT_LAGS,
        data_mean:   float | None = None,
        data_std:    float | None = None,
        path_steps=150,
        deterministic: bool = False,
        straight_bias: float = 0.75,
        enforce_blocks: bool = True,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
    ):
        self.lags = tuple(int(l) for l in lags)
        if not self.lags:
            raise ValueError("lags must be non-empty")
        self.max_lag = max(self.lags)
        self.path_steps = path_steps
        self.deterministic = deterministic
        self.straight_bias = straight_bias
        self.enforce_blocks = enforce_blocks
        self.block_threshold = block_threshold

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        arr = data[split]                                   # (H, W, 2, N)
        H, W, _, N = arr.shape
        if N <= self.max_lag:
            raise ValueError(f"split {split} has only {N} frames; need > {self.max_lag}")

        arr_t = np.transpose(arr, (3, 2, 0, 1)).astype(np.float32)   # (N, 2, H, W)
        self.land_mask = torch.from_numpy(np.isnan(arr[:, :, 0, 0]))  # (H, W) bool
        self.fields = torch.from_numpy(np.nan_to_num(arr_t, nan=0.0))  # (N, 2, H, W)

        self.data_mean = data_mean
        self.data_std = data_std
        if data_mean is not None and data_std is not None:
            self.fields = (self.fields - data_mean) / max(float(data_std), 1e-8)
            self.fields[:, :, self.land_mask] = 0.0          # re-zero land

        # The split is a concatenation of contiguous time-blocks (not one
        # continuous timeline).  A prior frame (f - L) is only physically valid
        # when it lies in the SAME block as the target f.  Label blocks and keep
        # only targets whose every prior is in-block.
        self.block_ids = detect_block_ids(
            self.fields.cpu().numpy(), self._ocean_for_blocks(),
            threshold=self.block_threshold,
        )
        if self.enforce_blocks:
            self.valid = valid_block_targets(self.block_ids, self.lags)
        else:
            # Naive: every frame with an in-array prior, ignoring boundaries.
            self.valid = np.arange(self.max_lag, N, dtype=np.int64)
        if len(self.valid) == 0:
            raise ValueError(
                f"split {split} has no valid block-respecting targets for "
                f"lags={self.lags}; blocks are too short."
            )

        # Geometry is static — compute once.
        self.geom = geometry_channels(self.land_mask)        # (3, H, W)

        self._land_np = self.land_mask.cpu().numpy()

    @staticmethod
    def compute_stats(pickle_path: str, split: int = 0) -> tuple[float, float]:
        """Return (mean, std) of ocean-cell values — same as OceanCurrentDataset."""
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
        priors = torch.cat([self.fields[f - L] for L in self.lags], dim=0)  # (2*nlags, H, W)

        seed = idx if self.deterministic else None
        rng = np.random.default_rng(seed)
        n_steps = self._sample_path_steps(rng)
        path_mask = biased_walk_path(
            self._land_np, n_steps=n_steps, seed=seed,
            straight_bias=self.straight_bias,
        )

        obs = observation_channels(target, path_mask)        # (3, H, W)
        cond = assemble_cond(obs, priors, self.geom)         # (C, H, W)
        return {"target": target, "cond": cond}
