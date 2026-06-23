"""
Build the chronological conditioning dataset from the raw .mat export.

The original ``data_divfree.pickle`` was a list of 3 split arrays carved into
short, time-scattered blocks.  Because each block averaged only ~21 frames —
shorter than the 25-hour conditioning look-back — ~80% of validation/test
targets had no real temporal prior (the prior fell across a block boundary into
an unrelated time).

The ``.mat`` export (``ramhead_dataset.mat``) is the SAME ocean-current record
but in *true chronological order*: 17 040 hourly frames with a perfectly
uniform 1-hour cadence and no gaps.  We therefore store ONE continuous,
divergence-free field array and define train/val/test as index lists of TARGET
frames into it.  Every target's 13 h / 25 h priors are looked up in the full
continuous array, so they are ALWAYS the genuine earlier field (we lose only
the first ``max(lags)`` frames of the whole record plus thin guard bands).

Output format (``Datasets/data_divfree_chrono.pickle``)::

    {
      "format":     "chrono_v1",
      "fields":     float32 (N, 2, H, W),  NaN at land  (divergence-free)
      "ocean_time": float64 (N,),          MATLAB datenum (hourly)
      "land_mask":  bool    (H, W),        True = land
      "splits":     {"train": int64[], "val": int64[], "test": int64[]},
                    # TARGET frame indices into `fields`
      "lags":       [13, 25],
      "data_mean":  0.0,                   # std-only (angle-preserving)
      "data_std":   float,                 # std of TRAIN-target ocean cells
      "meta":       {... build parameters ...},
    }

Priors are NOT materialised (that would triple the size and lock the lag set);
they are ``fields[idx - lag]`` and are documented to be valid for any
``idx`` in a split's index list.

Usage (from workspace root)::

    python utils/build_chrono_dataset.py \
        --mat Datasets/ramhead_dataset.mat \
        --out Datasets/data_divfree_chrono.pickle \
        --n_iter 20 --block_size 336 --guard 48 --verify
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from scipy.io import loadmat

_here  = os.path.dirname(os.path.abspath(__file__))
_model = os.path.join(_here, "..", "DDPM", "model")
sys.path.insert(0, _here)
sys.path.insert(0, _model)

from divfree_projection import divergence as compute_divergence, leray_project


# ---------------------------------------------------------------------------
# Divergence-free projection  (identical method to utils/project_dataset.py)
# ---------------------------------------------------------------------------

def project_fields(arr_t, land_mask_np, batch, n_iter, verify):
    """
    Project (N, 2, H, W) fields onto the divergence-free manifold.

    Reproduces the exact procedure that created ``data_divfree.pickle``:
    ``n_iter`` POCS iterations alternating spectral Leray projection and
    land-zeroing.

    Args:
        arr_t:        (N, 2, H, W) float32 torch tensor, land cells = 0.
        land_mask_np: (H, W) bool, True = land.
        batch:        batch size for projection.
        n_iter:       POCS iterations (20 matches the original).
        verify:       if True, measure mean|div| before/after.

    Returns:
        (N, 2, H, W) float32 tensor, divergence-free, land cells = 0.
    """
    N = arr_t.shape[0]
    ocean_mask_t = torch.from_numpy(~land_mask_np)
    land_mask_t  = torch.from_numpy(land_mask_np)

    out_t = torch.zeros_like(arr_t)
    before_divs, after_divs = [], []

    for start in range(0, N, batch):
        end = min(start + batch, N)
        x   = arr_t[start:end].clone()

        if verify:
            before_divs.append(
                compute_divergence(x, ocean_mask_t)[:, ocean_mask_t].abs().mean().item()
            )

        for _ in range(n_iter):
            x = leray_project(x, ocean_mask_t)
            x[:, :, land_mask_t] = 0.0

        if verify:
            after_divs.append(
                compute_divergence(x, ocean_mask_t)[:, ocean_mask_t].abs().mean().item()
            )

        out_t[start:end] = x
        print(f"\r  projecting: {end}/{N} ({100 * end // N}%)", end="", flush=True)

    print()
    if verify:
        print(f"    before mean|div|: {np.mean(before_divs):.6f}")
        print(f"    after  mean|div|: {np.mean(after_divs):.6f}")
    return out_t


# ---------------------------------------------------------------------------
# Chronological split with guard bands
# ---------------------------------------------------------------------------

# Block -> split assignment within each rotation of `_ROTATION` blocks.
_TRAIN, _VAL, _TEST = 0, 1, 2
_ROTATION = (_TRAIN, _TRAIN, _TRAIN, _TRAIN, _TRAIN, _VAL, _TEST)  # 5/1/1 ≈ 71/14/14
_SPLIT_NAMES = {_TRAIN: "train", _VAL: "val", _TEST: "test"}


def build_splits(N, block_size, guard, max_lag):
    """
    Assign every frame to train/val/test by rotating contiguous blocks, then
    keep only target frames that (a) have a full look-back and (b) are at least
    ``guard`` frames away from any frame of a DIFFERENT split.

    The guard band (>= max_lag) does two things at once:
      * decorrelates val/test targets from train targets (no near-duplicate
        leakage — adjacent frames correlate ~0.95);
      * absorbs all cross-boundary prior reach, so no kept target's field is
        ever used as a prior for a different split's target.

    Args:
        N:          number of frames.
        block_size: frames per rotation block.
        guard:      guard-band half-width in frames (>= max_lag recommended).
        max_lag:    largest conditioning lag (frames with index < max_lag are
                    dropped — they have no prior).

    Returns:
        dict name -> int64 array of valid target indices,
        and the per-frame split-label array (for diagnostics).
    """
    block_id    = np.arange(N) // block_size
    frame_label = np.array([_ROTATION[b % len(_ROTATION)] for b in block_id], dtype=np.int64)

    # Distance to the nearest frame of a different split.  A frame is "tainted"
    # (dropped) if a different-split frame lies within `guard`.
    diff_boundary = np.zeros(N, dtype=bool)
    diff_boundary[:-1] |= frame_label[:-1] != frame_label[1:]
    diff_boundary[1:]  |= frame_label[1:]  != frame_label[:-1]
    # indices where a split change happens (between i and i+1)
    change_idx = np.where(frame_label[:-1] != frame_label[1:])[0]

    tainted = np.zeros(N, dtype=bool)
    for c in change_idx:
        lo = max(0, c - guard + 1)
        hi = min(N, c + 1 + guard)
        tainted[lo:hi] = True

    splits = {}
    for label, name in _SPLIT_NAMES.items():
        sel = (frame_label == label) & (~tainted)
        sel[:max_lag] = False                      # no look-back available
        splits[name] = np.where(sel)[0].astype(np.int64)
    return splits, frame_label


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset(fields_np, land_mask_np, splits, frame_label, lags, guard):
    """Print integrity checks: coverage, prior validity, leakage separation."""
    N = fields_np.shape[0]
    ocean = ~land_mask_np
    max_lag = max(lags)
    print("\n===== VERIFICATION =====")
    print(f"Total frames: {N}")

    # 1. sizes + timeline coverage
    for name, idx in splits.items():
        if len(idx) == 0:
            print(f"  {name:5s}: EMPTY")
            continue
        span = f"[{idx.min()}, {idx.max()}]"
        print(f"  {name:5s}: {len(idx):6d} targets  span {span}  "
              f"({100 * len(idx) / N:.1f}% of frames)")

    # 2. no index collisions between splits
    all_idx = np.concatenate([splits[n] for n in splits])
    assert len(all_idx) == len(np.unique(all_idx)), "split target indices overlap!"
    print("  [ok] no target index appears in more than one split")

    # 3. prior validity — every prior in-range and genuinely earlier
    for name, idx in splits.items():
        if len(idx) == 0:
            continue
        for L in lags:
            assert (idx - L >= 0).all(), f"{name}: prior lag {L} underflows"
    print(f"  [ok] all priors in-range for lags {tuple(lags)}")

    # 4. leakage: min temporal distance between different-split targets
    def min_cross_distance(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.inf
        b_sorted = np.sort(b)
        pos = np.searchsorted(b_sorted, a)
        best = np.full(len(a), np.inf)
        for k, p in enumerate(pos):
            if p < len(b_sorted):
                best[k] = min(best[k], abs(b_sorted[p] - a[k]))
            if p > 0:
                best[k] = min(best[k], abs(a[k] - b_sorted[p - 1]))
        return best.min()

    dtv = min_cross_distance(splits["val"], splits["train"])
    dtt = min_cross_distance(splits["test"], splits["train"])
    dvt = min_cross_distance(splits["val"], splits["test"])
    print(f"  min |val-train| = {dtv:.0f} frames, |test-train| = {dtt:.0f}, "
          f"|val-test| = {dvt:.0f}  (guard = {guard})")
    assert min(dtv, dtt, dvt) >= guard, "leakage: split targets closer than guard!"
    print(f"  [ok] all cross-split targets >= guard ({guard}h) apart")

    # 5. correlation decay — what guard buys us
    flat = fields_np[:, :, ocean].reshape(N, -1).astype(np.float64)
    flat = np.nan_to_num(flat)
    flat -= flat.mean(1, keepdims=True)
    flat /= (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    for k in (1, 13, 25, guard, 2 * guard):
        if k < N:
            c = np.einsum("ij,ij->i", flat[:-k], flat[k:]).mean()
            print(f"  mean corr(frame, frame+{k:3d}h) = {c:+.3f}")

    # 6. prior realism — corr(target, its 25h prior) should be high & positive
    for L in lags:
        c = np.einsum("ij,ij->i", flat[L:], flat[:-L]).mean()
        print(f"  mean corr(target, prior-{L}h) = {c:+.3f}  (real temporal prior)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="Datasets/ramhead_dataset.mat")
    p.add_argument("--out", default="Datasets/data_divfree_chrono.pickle")
    p.add_argument("--n_iter",     type=int, default=20, help="POCS projection iterations")
    p.add_argument("--batch",      type=int, default=64)
    p.add_argument("--block_size", type=int, default=336, help="frames per split block (336 = 14 days)")
    p.add_argument("--guard",      type=int, default=48,  help="guard band half-width (frames)")
    p.add_argument("--lags",       default="13,25")
    p.add_argument("--verify",     action="store_true")
    args = p.parse_args()

    lags = tuple(int(x) for x in args.lags.split(","))
    max_lag = max(lags)
    if args.guard < max_lag:
        print(f"WARNING: guard ({args.guard}) < max_lag ({max_lag}); "
              f"raising guard to {max_lag} to absorb prior reach.")
        args.guard = max_lag

    print(f"Loading {args.mat} ...")
    m = loadmat(args.mat)
    u = m["u"].astype(np.float32)          # (H, W, N)
    v = m["v"].astype(np.float32)          # (H, W, N)
    ocean_time = m["ocean_time"].ravel().astype(np.float64)
    H, W, N = u.shape
    print(f"Grid {H}x{W}, frames {N}, time {ocean_time[0]:.4f}..{ocean_time[-1]:.4f}")

    land_mask_np = np.isnan(u[:, :, 0])    # (H, W) — constant across time
    print(f"Land cells: {land_mask_np.sum()}  ocean cells: {(~land_mask_np).sum()}")

    # (H, W, N) x2 -> (N, 2, H, W), land -> 0 for projection
    arr = np.stack([u, v], axis=2)                       # (H, W, 2, N)
    arr_t = torch.from_numpy(
        np.nan_to_num(np.transpose(arr, (3, 2, 0, 1)), nan=0.0)
    )                                                    # (N, 2, H, W)
    del u, v, arr

    print(f"Projecting {N} frames ({args.n_iter} POCS iters) ...")
    fields = project_fields(arr_t, land_mask_np, args.batch, args.n_iter, args.verify)
    del arr_t

    # store with NaN at land (matches data_divfree convention)
    fields_np = fields.numpy().copy()
    fields_np[:, :, land_mask_np] = np.nan
    del fields

    splits, frame_label = build_splits(N, args.block_size, args.guard, max_lag)

    if args.verify:
        verify_dataset(fields_np, land_mask_np, splits, frame_label, lags, args.guard)

    # std-only normalization stat (mean forced to 0, angle-preserving) from
    # TRAIN-target ocean cells only — never peeks at val/test.
    train_idx = splits["train"]
    ocean = ~land_mask_np
    train_vals = fields_np[train_idx][:, :, ocean]       # (n_train, 2, n_ocean)
    train_vals = train_vals[~np.isnan(train_vals)]
    data_std = float(train_vals.std())
    print(f"\nstd-only normalization: mean=0.0  std={data_std:.5f}")

    out = {
        "format":     "chrono_v1",
        "fields":     fields_np.astype(np.float32),
        "ocean_time": ocean_time,
        "land_mask":  land_mask_np,
        "splits":     splits,
        "lags":       list(lags),
        "data_mean":  0.0,
        "data_std":   data_std,
        "meta": {
            "source_mat": os.path.basename(args.mat),
            "n_iter":     args.n_iter,
            "block_size": args.block_size,
            "guard":      args.guard,
            "rotation":   _ROTATION,
            "grid":       [H, W],
            "n_frames":   N,
        },
    }

    print(f"Saving -> {args.out} ...")
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Done. {size_mb:.0f} MB  | train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
