"""
Build the chronological conditioning dataset from full_dataset.mat (full island,
242×329 grid, 18192 hourly frames).

Mirrors build_chrono_dataset.py but handles the differences in full_dataset.mat:
  - HDF5 v7.3 format  → h5py (not scipy.io)
  - fields stored as (N, H, W) → transposed to (N, 2, H, W)
  - land mask from `mask` key (0 = land) instead of NaN detection

Memory: ~23 GB of float32 fields.  Run on the GPU server, not a laptop.

Output (Datasets/pickles/full_island_chrono.pickle) has the same chrono_v1
format as data_divfree_chrono.pickle and is a drop-in replacement for all
training / inference scripts — just pass --pickle <new_path>.

Usage (from workspace root, GPU server):
    python Utils/build_full_island_dataset.py \
        --mat  Datasets/full_datasets/full_dataset.mat \
        --out  Datasets/pickles/full_island_chrono.pickle \
        --n_iter 20 --batch 32 --verify
"""

import argparse
import os
import pickle
import sys

import h5py
import numpy as np
import torch

_here  = os.path.dirname(os.path.abspath(__file__))
_model = os.path.join(_here, "..", "DDPM", "model")
sys.path.insert(0, _here)
sys.path.insert(0, _model)

from divfree_projection import divergence as compute_divergence, leray_project

# Reuse split logic from build_chrono_dataset — same rotation scheme.
_TRAIN, _VAL, _TEST = 0, 1, 2
_ROTATION = (_TRAIN, _TRAIN, _TRAIN, _TRAIN, _TRAIN, _VAL, _TEST)
_SPLIT_NAMES = {_TRAIN: "train", _VAL: "val", _TEST: "test"}


def build_splits(N, block_size, guard, max_lag):
    block_id    = np.arange(N) // block_size
    frame_label = np.array([_ROTATION[b % len(_ROTATION)] for b in block_id], dtype=np.int64)
    change_idx  = np.where(frame_label[:-1] != frame_label[1:])[0]
    tainted     = np.zeros(N, dtype=bool)
    for c in change_idx:
        tainted[max(0, c - guard + 1):min(N, c + 1 + guard)] = True
    splits = {}
    for label, name in _SPLIT_NAMES.items():
        sel = (frame_label == label) & (~tainted)
        sel[:max_lag] = False
        splits[name] = np.where(sel)[0].astype(np.int64)
    return splits, frame_label


def project_fields(f_h5, land_mask_np, N, H, W, batch, n_iter, verify, device):
    """
    Stream-project fields from the open HDF5 file in chunks to avoid loading
    all 23 GB into RAM at once.

    f_h5: open h5py.File
    Returns projected fields as a float32 numpy array (N, 2, H, W) with NaN at land.
    """
    ocean_mask_t = torch.from_numpy(~land_mask_np).to(device)
    land_mask_t  = torch.from_numpy(land_mask_np).to(device)

    out = np.empty((N, 2, H, W), dtype=np.float32)
    out[:] = np.nan
    before_divs, after_divs = [], []

    for start in range(0, N, batch):
        end = min(start + batch, N)
        u_b = torch.from_numpy(f_h5["us"][start:end].astype(np.float32)).to(device)  # (B, H, W)
        v_b = torch.from_numpy(f_h5["vs"][start:end].astype(np.float32)).to(device)
        x   = torch.stack([u_b, v_b], dim=1)                                          # (B, 2, H, W)
        x[:, :, land_mask_t] = 0.0
        del u_b, v_b

        if verify:
            before_divs.append(
                compute_divergence(x.cpu(), ocean_mask_t.cpu())[:, ocean_mask_t.cpu()].abs().mean().item()
            )

        for _ in range(n_iter):
            x = leray_project(x, ocean_mask_t)
            x[:, :, land_mask_t] = 0.0

        if verify:
            after_divs.append(
                compute_divergence(x.cpu(), ocean_mask_t.cpu())[:, ocean_mask_t.cpu()].abs().mean().item()
            )

        x_np = x.cpu().numpy()
        x_np[:, :, land_mask_np] = np.nan
        out[start:end] = x_np
        del x

        print(f"\r  projecting: {end}/{N} ({100 * end // N}%)", end="", flush=True)

    print()
    if verify:
        print(f"    before mean|div|: {np.mean(before_divs):.6f}")
        print(f"    after  mean|div|: {np.mean(after_divs):.6f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat",        default="Datasets/full_datasets/full_dataset.mat")
    p.add_argument("--out",        default="Datasets/pickles/full_island_chrono.pickle")
    p.add_argument("--n_iter",     type=int, default=20)
    p.add_argument("--batch",      type=int, default=32,
                   help="Frames per projection batch. Reduce if OOM.")
    p.add_argument("--block_size", type=int, default=336,
                   help="Frames per split block (336 = 14 days).")
    p.add_argument("--guard",      type=int, default=48)
    p.add_argument("--lags",       default="13,25")
    p.add_argument("--verify",     action="store_true")
    p.add_argument("--device",     default=None,
                   help="cuda | mps | cpu (auto-detect if unset).")
    args = p.parse_args()

    lags    = tuple(int(x) for x in args.lags.split(","))
    max_lag = max(lags)
    if args.guard < max_lag:
        args.guard = max_lag
        print(f"guard raised to {max_lag} to cover max lag")

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Opening {args.mat} ...")
    with h5py.File(args.mat, "r") as f:
        N, H, W = f["us"].shape
        ocean_time = f["ocean_time"][:].ravel().astype(np.float64)
        mask_np    = f["mask"][()].astype(np.float32)       # (H, W) 1=ocean 0=land
        land_mask_np = (mask_np < 0.5)                      # True = land

        print(f"Grid {H}×{W}, frames {N}  "
              f"(land={land_mask_np.sum()}  ocean={(~land_mask_np).sum()})")
        est_gb = N * 2 * H * W * 4 / 1e9
        print(f"Estimated output size: {est_gb:.1f} GB  (batch={args.batch})")

        print(f"Projecting {N} frames onto divergence-free manifold "
              f"({args.n_iter} POCS iters) ...")
        fields_np = project_fields(
            f, land_mask_np, N, H, W, args.batch, args.n_iter, args.verify, device)

    splits, frame_label = build_splits(N, args.block_size, args.guard, max_lag)
    for name, idx in splits.items():
        print(f"  {name:5s}: {len(idx):6d} targets")

    # std-only normalization from train targets only
    train_idx = splits["train"]
    ocean     = ~land_mask_np
    train_vals = fields_np[train_idx][:, :, ocean]
    train_vals = train_vals[~np.isnan(train_vals)]
    data_std   = float(train_vals.std())
    print(f"std-only normalization: mean=0.0  std={data_std:.5f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
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
    print(f"Done. {size_mb:.0f} MB  |  "
          f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")


if __name__ == "__main__":
    main()
