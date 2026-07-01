"""
Quick-look visualiser for a raw .mat ocean-current export.

Renders ONE time frame as a quiver plot coloured by current speed with land in
black — the same visual style used throughout the project — so you can eyeball a
dataset before building anything from it.

Supports both layouts we have:
  * ramhead_dataset.mat  (MATLAB <= v7) : u,v  (H, W, N),  land = NaN
  * full_dataset.mat     (MATLAB v7.3)  : us,vs (N, H, W), mask (1=water,0=land)

Usage (from workspace root)::

    python utils/plot_mat_sample.py \
        --mat   Datasets/full_dataset.mat \
        --frame 0 \
        --out   Datasets/full_dataset_sample.png
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_frame(mat_path, frame):
    """Return (u, v, land_mask) for one frame; u,v are (H, W) with NaN at land."""
    try:
        from scipy.io import loadmat
        m = loadmat(mat_path)
        # ramhead layout: u,v are (H, W, N), land already NaN.
        u = m["u"][:, :, frame].astype(np.float32)
        v = m["v"][:, :, frame].astype(np.float32)
        land = np.isnan(u)
        return u, v, land
    except (NotImplementedError, ValueError):
        # MATLAB v7.3 (HDF5) layout: us,vs are (N, H, W), explicit mask.
        import h5py
        with h5py.File(mat_path, "r") as f:
            u = np.asarray(f["us"][frame], dtype=np.float32)
            v = np.asarray(f["vs"][frame], dtype=np.float32)
            mask = np.asarray(f["mask"], dtype=np.float32)   # 1=water, 0=land
        land = mask == 0
        u[land] = np.nan
        v[land] = np.nan
        return u, v, land


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat",   default="Datasets/full_dataset.mat")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--step",  type=int, default=4,
                   help="quiver subsampling stride (larger = sparser arrows)")
    p.add_argument("--out",   default="Datasets/full_dataset_sample.png")
    args = p.parse_args()

    u, v, land = load_frame(args.mat, args.frame)
    H, W = u.shape
    ocean = ~land
    n_ocean = int(ocean.sum())
    print(f"{os.path.basename(args.mat)}  frame {args.frame}  grid {H}x{W}  "
          f"ocean cells {n_ocean} ({100 * n_ocean / (H * W):.1f}%)")

    speed = np.sqrt(u ** 2 + v ** 2)
    vmax = float(np.nanpercentile(speed[ocean], 98)) if n_ocean else 1.0
    print(f"speed: mean {np.nanmean(speed):.3f}  98th pct {vmax:.3f}  "
          f"max {np.nanmax(speed):.3f}")

    fig, ax = plt.subplots(figsize=(W / 22 + 2, H / 22 + 1))
    ax.imshow(
        land, origin="lower",
        cmap=mcolors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="equal", zorder=0,
    )
    s = args.step
    yq, xq = np.mgrid[0:H:s, 0:W:s]
    uq, vq = u[::s, ::s], v[::s, ::s]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    sel = ~np.isnan(uq) & ~land[::s, ::s]
    q = ax.quiver(
        xq[sel], yq[sel], uq[sel], vq[sel], mq[sel],
        cmap="cool", clim=(0, vmax), scale=18, width=0.002, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(f"{os.path.basename(args.mat)} — frame {args.frame}", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
