"""
Compute divergence statistics for every field in data.pickle.

Reports mean |div|, max |div|, and % of ocean cells exceeding a threshold
for each split (train / val / test) and overall.

Usage (from workspace root):
    python utils/check_dataset_divergence.py
    python utils/check_dataset_divergence.py --pickle data.pickle --threshold 0.01
"""

import argparse
import os
import sys

import numpy as np
import torch

# ---- make divfree_projection importable from any working directory ----
_here  = os.path.dirname(os.path.abspath(__file__))
_model = os.path.join(_here, "..", "DDPM", "model")
sys.path.insert(0, _here)
sys.path.insert(0, _model)

from divfree_projection import divergence as compute_divergence  # noqa: E402
from dataset import OceanCurrentDataset                          # noqa: E402


def split_stats(split_idx: int, pickle_path: str, threshold: float, batch: int = 64):
    ds         = OceanCurrentDataset(pickle_path, split=split_idx)
    ocean_mask = ~ds.land_mask   # (H, W) True = ocean

    all_mean_div = []
    all_max_div  = []
    all_pct      = []

    N = len(ds)
    for start in range(0, N, batch):
        end   = min(start + batch, N)
        x     = ds.data[start:end]                    # (B, 2, H, W)
        div   = compute_divergence(x, ocean_mask)     # (B, H, W)

        # only ocean cells
        ocean_vals = div[:, ocean_mask]               # (B, ocean_cells)
        abs_vals   = ocean_vals.abs()

        all_mean_div.append(abs_vals.mean(dim=1))     # (B,)
        all_max_div.append(abs_vals.max(dim=1).values)
        all_pct.append((abs_vals > threshold).float().mean(dim=1))

    mean_div = torch.cat(all_mean_div)
    max_div  = torch.cat(all_max_div)
    pct      = torch.cat(all_pct)

    return {
        "n":           N,
        "mean_div":    float(mean_div.mean()),
        "std_div":     float(mean_div.std()),
        "median_div":  float(mean_div.median()),
        "max_div":     float(max_div.max()),
        "pct_exceed":  float(pct.mean()) * 100,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",    default="data.pickle")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="divergence threshold for 'pct exceed' metric")
    p.add_argument("--batch",     type=int, default=64)
    args = p.parse_args()

    split_names = ["Train", "Val  ", "Test "]

    print(f"\nDivergence audit: {args.pickle}")
    print(f"Threshold for '% exceed': {args.threshold}")
    print()

    hdr = f"  {'Split':<8}  {'N':>5}  {'mean|div|':>10}  {'std':>8}  "  \
          f"{'median':>8}  {'max':>8}  {'%>thresh':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_means = []
    for i, name in enumerate(split_names):
        s = split_stats(i, args.pickle, args.threshold, args.batch)
        all_means.append(s["mean_div"])
        print(
            f"  {name:<8}  {s['n']:>5}  {s['mean_div']:>10.6f}  "
            f"{s['std_div']:>8.6f}  {s['median_div']:>8.6f}  "
            f"{s['max_div']:>8.4f}  {s['pct_exceed']:>8.2f}%"
        )

    print()
    print(f"  Overall mean |div|: {np.mean(all_means):.6f}")
    print()


if __name__ == "__main__":
    main()
