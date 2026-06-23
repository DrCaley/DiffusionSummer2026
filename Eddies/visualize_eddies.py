"""
Visualise ocean current fields with detected eddies marked.

Reads the eddy catalog produced by find_eddies.py and overlays eddy
centre markers on quiver plots of the velocity field.

Markers:
  Red  dot  = cyclonic eddy       (counter-clockwise)
  Blue dot  = anticyclonic eddy   (clockwise)

Usage (from workspace root)
---------------------------
  # Show sample 0 from train split:
  python eddies/visualize_eddies.py

  # Pick a specific sample and split:
  python eddies/visualize_eddies.py --split val --sample 42

  # Show the N samples with the most eddies:
  python eddies/visualize_eddies.py --mode top --topn 6

  # Show a grid of random eddy-containing samples:
  python eddies/visualize_eddies.py --mode random --topn 9 --seed 7

  # Re-generate catalog first if it doesn't exist:
  python eddies/find_eddies.py
  python eddies/visualize_eddies.py --sample 0
"""

import argparse
import json
import os
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_field(pickle_path: str, split: int, sample_index: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (u, v, land_mask):
      u, v        : (H, W) float32
      land_mask   : (H, W) bool  (True = land)
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    arr       = data[split]                    # (H, W, 2, N)
    land_mask = np.isnan(arr[:, :, 0, 0])
    u = np.nan_to_num(arr[:, :, 0, sample_index], nan=0.0)
    v = np.nan_to_num(arr[:, :, 1, sample_index], nan=0.0)
    return u, v, land_mask


def _plot_one(ax: plt.Axes,
              u: np.ndarray,
              v: np.ndarray,
              land_mask: np.ndarray,
              eddies: list[dict],
              title: str):
    """Draw quiver + land overlay + eddy markers on ax."""
    H, W = u.shape

    # Transpose for display: (H=94, W=44) → (44, 94)  so x=east, y=north
    ut = u.T
    vt = v.T
    lt = land_mask.T

    speed = np.hypot(ut, vt)
    speed_masked = np.ma.masked_where(lt, speed)

    # Grid
    X, Y = np.meshgrid(np.arange(H), np.arange(W))

    # Land background
    land_rgba = np.zeros((*lt.shape, 4))
    land_rgba[lt] = [0, 0, 0, 1]
    ax.imshow(land_rgba, origin="lower",
              extent=[-0.5, H - 0.5, -0.5, W - 0.5], aspect="auto", zorder=1)

    # Quiver — full resolution so eddy circulation is visible
    mask2d = ~np.ma.getmaskarray(speed_masked)
    q = ax.quiver(X[mask2d], Y[mask2d], ut[mask2d], vt[mask2d],
                  speed_masked[mask2d],
                  cmap="cool", scale=6.0, scale_units="inches",
                  clim=[0, 0.3], width=0.002, zorder=2)
    plt.colorbar(q, ax=ax, label="Speed (m/s)", shrink=0.7)

    # Eddy markers — large dot + circle outline + label
    for eddy in eddies:
        px = eddy["center_x"]   # along H (east–west) = plot x after transpose
        py = eddy["center_y"]   # along W (north–south) = plot y after transpose
        color = "red" if eddy["type"] == "cyclonic" else "blue"
        # Filled dot
        ax.scatter(px, py, s=300, c=color, marker="o",
                   edgecolors="white", linewidths=1.5, zorder=6)
        # Dashed circle showing approximate eddy radius
        radius = np.sqrt(eddy["n_cells"] / np.pi)
        circle = plt.Circle((px, py), radius, color=color,
                             fill=False, linewidth=1.5, linestyle="--", zorder=5)
        ax.add_patch(circle)
        # Label
        ax.text(px + radius + 0.5, py,
                f"{eddy['type'][0].upper()}  ω={eddy['mean_vorticity']:.3f}  circ={eddy.get('circ_ratio',0):.2f}",
                fontsize=8, color=color, fontweight="bold",
                va="center", zorder=7)

    # Legend
    n_cyc  = sum(1 for e in eddies if e["type"] == "cyclonic")
    n_anti = sum(1 for e in eddies if e["type"] == "anticyclonic")
    handles = [
        mpatches.Patch(facecolor="black", label="Land"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=10, label=f"Cyclonic ({n_cyc})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
                   markersize=10, label=f"Anticyclonic ({n_anti})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (east–west grid index)")
    ax.set_ylabel("Y (north–south grid index)")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

SPLIT_ID = {"train": 0, "val": 1, "test": 2}


def _get_eddies_for(catalog: list[dict], split_name: str, sample_index: int
                    ) -> list[dict]:
    return [e for e in catalog
            if e["split"] == split_name and e["sample_index"] == sample_index]


def mode_single(args, catalog, pickle_path):
    """Plot a single sample."""
    split_name = args.split
    idx        = args.sample
    u, v, land = _load_field(pickle_path, SPLIT_ID[split_name], idx)
    eddies     = _get_eddies_for(catalog, split_name, idx)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    _plot_one(ax, u, v, land, eddies,
              f"{split_name} sample {idx} — {len(eddies)} eddy/ies detected")
    plt.tight_layout()

    out = args.out or f"eddies/eddy_{split_name}_{idx}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def mode_top(args, catalog, pickle_path):
    """Grid of samples with the most eddies."""
    from collections import Counter
    rng = np.random.default_rng(args.seed)

    # Count eddies per (split, sample)
    counts = Counter((e["split"], e["sample_index"]) for e in catalog)
    top    = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    if args.mode == "random":
        # Random sample from those that have at least one eddy
        keys = list(counts.keys())
        rng.shuffle(keys)
        top = [(k, counts[k]) for k in keys]

    n     = min(args.topn, len(top))
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols / 3, 6 * nrows))
    axes = np.array(axes).flatten()

    for i, ((split_name, sample_index), count) in enumerate(top[:n]):
        u, v, land = _load_field(pickle_path, SPLIT_ID[split_name], sample_index)
        eddies     = _get_eddies_for(catalog, split_name, sample_index)
        _plot_one(axes[i], u, v, land, eddies,
                  f"{split_name}[{sample_index}]  {count} eddies")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"{'Top' if args.mode == 'top' else 'Random'} {n} eddy-containing samples",
        fontsize=13
    )
    plt.tight_layout()

    out = args.out or f"eddies/eddy_{'top' if args.mode == 'top' else 'random'}_{n}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="eddies/eddy_catalog.json",
                   help="Eddy catalog JSON produced by find_eddies.py")
    p.add_argument("--pickle",  default="Datasets/data.pickle")
    p.add_argument("--mode",    default="single",
                   choices=["single", "top", "random"],
                   help="single: one sample | top: N samples with most eddies | "
                        "random: N random eddy-containing samples")
    p.add_argument("--split",   default="train",
                   choices=["train", "val", "test"],
                   help="Split to use (only for --mode single)")
    p.add_argument("--sample",  type=int, default=0,
                   help="Sample index within split (only for --mode single)")
    p.add_argument("--topn",    type=int, default=6,
                   help="Number of samples to show in top/random mode")
    p.add_argument("--seed",    type=int, default=42,
                   help="Random seed for --mode random")
    p.add_argument("--out",     default=None,
                   help="Output image path (auto-named if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.catalog):
        print(f"ERROR: catalog not found: {args.catalog}")
        print("Run:  python eddies/find_eddies.py  first.")
        sys.exit(1)

    with open(args.catalog) as f:
        data    = json.load(f)
    catalog = data["eddies"]
    meta    = data["metadata"]

    print(f"Catalog: {args.catalog}")
    print(f"  Total eddies: {meta['total_eddies']}  "
          f"(cyclonic={meta['cyclonic']}, anticyclonic={meta['anticyclonic']})")
    print(f"  OW threshold: {meta['ow_threshold']} × std(W), "
          f"min_cells={meta['min_cells']}")
    print()

    if args.mode == "single":
        mode_single(args, catalog, args.pickle)
    else:
        mode_top(args, catalog, args.pickle)


if __name__ == "__main__":
    main()
