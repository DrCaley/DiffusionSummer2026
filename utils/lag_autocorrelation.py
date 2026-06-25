"""Temporal autocorrelation of the ocean-current field vs lag (1..N hours).

Uses the chronological pickle (continuous, true 1-hour cadence) so a lag of L
frames == L hours.  For each lag L we correlate every frame t with frame t-L
over ocean cells, averaged over all valid t, for two quantities:

  * whole-field (u,v)  -> dominated by the tidal sloshing
  * vorticity  w = dv/dx - du/dy  -> the rotational / eddy structure (north star)

Pearson correlation pooled over (ocean cells x components x frame-pairs).
Saves a PNG line plot with the same-tidal-phase peaks (13 h, 25 h) marked.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "DDPM", "model"))


def vorticity(fields, ocean):
    """w = dv/dx - du/dy via central differences.  fields (N,2,H,W) -> (N,H,W)."""
    u = fields[:, 0:1]
    v = fields[:, 1:2]
    # physical axes: H = x (east-west), W = y (north-south)  (see divfree_projection)
    kH = torch.tensor([[[[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]]]]) / 2.0  # d/dx
    kW = torch.tensor([[[[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]]]]) / 2.0  # d/dy
    dvdx = F.conv2d(v, kH, padding=1)
    dudy = F.conv2d(u, kW, padding=1)
    w = (dvdx - dudy).squeeze(1)
    return w * ocean


def pooled_corr(A, B, mask):
    """Pearson corr pooled over all masked entries of two stacks A,B (same shape)."""
    a = A[..., mask]
    b = B[..., mask]
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    ap.add_argument("--max_lag", type=int, default=100)
    ap.add_argument("--max_frames", type=int, default=2000,
                    help="cap on number of target frames sampled (speed)")
    ap.add_argument("--out", default="lag_autocorrelation.png")
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    fields = np.nan_to_num(np.asarray(data["fields"], dtype=np.float32), nan=0.0)
    land = np.asarray(data["land_mask"], dtype=bool)
    N = fields.shape[0]
    ocean_np = ~land
    ocean = torch.from_numpy(ocean_np)
    print(f"frames={N}  ocean cells={int(ocean_np.sum())}  max_lag={args.max_lag}")

    F_t = torch.from_numpy(fields)                       # (N,2,H,W)
    F_t = F_t * ocean[None, None]
    W_t = vorticity(F_t, ocean.float())                  # (N,H,W)

    lags = list(range(1, args.max_lag + 1))
    whole, vort = [], []
    for L in lags:
        # frame indices t (target) and t-L (prior); subsample for speed
        idx = np.arange(L, N)
        if idx.size > args.max_frames:
            idx = np.linspace(L, N - 1, args.max_frames).astype(int)
        idx_t = torch.from_numpy(idx)
        idx_p = torch.from_numpy(idx - L)
        whole.append(pooled_corr(F_t[idx_t], F_t[idx_p], ocean))
        vort.append(pooled_corr(W_t[idx_t], W_t[idx_p], ocean))
        if L % 10 == 0:
            print(f"  lag {L:3d}h  whole={whole[-1]:.3f}  vort={vort[-1]:.3f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    ax.plot(lags, whole, "-", color="tab:blue", lw=2,
            label="Whole-field (u,v)  — tidal")
    ax.plot(lags, vort, "-", color="tab:red", lw=2,
            label="Vorticity ω = ∂v/∂x − ∂u/∂y  — eddy / north star")

    for L in (13, 25):
        ax.axvline(L, color="gray", ls="--", lw=1, alpha=0.7)
        ax.text(L, ax.get_ylim()[1], f" {L}h", va="top", ha="left",
                fontsize=9, color="gray")
    # tidal-phase guides (semidiurnal ~12.42 h)
    for k in range(1, args.max_lag // 12 + 1):
        ax.axvline(12.42 * k, color="tab:green", ls=":", lw=0.8, alpha=0.35)

    ax.set_xlabel("Lag (hours prior)")
    ax.set_ylabel("Pearson correlation (ocean cells)")
    ax.set_title("Temporal autocorrelation of the ocean-current field vs lag\n"
                 "(green dotted = semidiurnal tidal cycle ~12.42 h multiples)")
    ax.set_xlim(1, args.max_lag)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"\nsaved -> {args.out}")
    # quick text summary at the canonical lags
    for L in (1, 13, 25, 37, 49):
        if L <= args.max_lag:
            print(f"  lag {L:3d}h: whole={whole[L-1]:.3f}  vort={vort[L-1]:.3f}")


if __name__ == "__main__":
    main()
