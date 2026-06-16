"""
Eddy detection across the ocean current dataset using the Okubo-Weiss parameter.

Background
----------
The Okubo-Weiss (OW) parameter W = sn² + ss² − ω²  decomposes the flow into:
  sn  = ∂u/∂x − ∂v/∂y   (normal strain)
  ss  = ∂v/∂x + ∂u/∂y   (shear strain)
  ω   = ∂v/∂x − ∂u/∂y   (vorticity)

W < 0  → rotation dominates → eddy core
W > 0  → strain dominates   → filaments / fronts

Cells with W < −OW_THRESHOLD * std(W) are labelled as eddy cores.
Connected components below that threshold are identified as individual eddies;
small components (< MIN_EDDY_CELLS) are discarded as noise.

The sign of the mean vorticity ω inside each component classifies the eddy:
  ω > 0  → cyclonic     (counter-clockwise in standard orientation)
  ω < 0  → anticyclonic (clockwise)

Grid convention (matches the rest of this project)
---------------------------------------------------
  Array axis 0  = H = 94   (east–west,   x-direction,  u channel)
  Array axis 1  = W = 44   (north–south, y-direction,  v channel)
  channel 0     = u  (east–west velocity)
  channel 1     = v  (north–south velocity)

Usage
-----
  # From workspace root:
  python eddies/find_eddies.py --pickle Datasets/data.pickle --out eddies/eddy_catalog.json

  # Scan only the test split and be verbose:
  python eddies/find_eddies.py --split 2 --verbose

  # Adjust sensitivity:
  python eddies/find_eddies.py --ow_threshold 0.1 --min_cells 5
"""

import argparse
import json
import os
import sys

import numpy as np
import pickle
from scipy.ndimage import label, center_of_mass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Physical computation
# ---------------------------------------------------------------------------

def _central_diff_x(arr: np.ndarray) -> np.ndarray:
    """∂arr/∂x  along axis-0 (H, east–west), central differences, zero at edges."""
    d = np.zeros_like(arr)
    d[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / 2.0
    return d


def _central_diff_y(arr: np.ndarray) -> np.ndarray:
    """∂arr/∂y  along axis-1 (W, north–south), central differences, zero at edges."""
    d = np.zeros_like(arr)
    d[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / 2.0
    return d


def okubo_weiss(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Okubo-Weiss parameter W and vorticity ω for a 2-D velocity field.

    Args:
        u: (H, W) east–west velocity component
        v: (H, W) north–south velocity component

    Returns:
        W   : (H, W) Okubo-Weiss parameter
        omega: (H, W) vorticity  ω = ∂v/∂x − ∂u/∂y
    """
    dudx = _central_diff_x(u)   # ∂u/∂x
    dudy = _central_diff_y(u)   # ∂u/∂y
    dvdx = _central_diff_x(v)   # ∂v/∂x
    dvdy = _central_diff_y(v)   # ∂v/∂y

    sn    = dudx - dvdy          # normal strain
    ss    = dvdx + dudy          # shear strain
    omega = dvdx - dudy          # vorticity

    W = sn**2 + ss**2 - omega**2
    return W, omega


def _circulation_ratio(
    u: np.ndarray,
    v: np.ndarray,
    cx: float,
    cy: float,
    comp: np.ndarray,
) -> float:
    """
    Compute the mean tangential / (tangential + radial) velocity ratio for the
    cells in `comp` around centroid (cx, cy).

    A true eddy has velocity vectors that wrap around the center:
      tangential component >> radial component  →  ratio close to 1.0

    A shear flow near a boundary has no consistent wrap-around:
      radial and tangential are similar magnitudes  →  ratio close to 0.5

    Returns a value in [0, 1].  We require > CIRC_THRESHOLD to keep the eddy.
    """
    rows, cols = np.where(comp)
    # Vector from centroid to each cell
    dx = cols - cx   # along axis-1 (W, north–south in array, y in display)
    dy = rows - cy   # along axis-0 (H, east–west in array, x in display)
    dist = np.hypot(dx, dy)
    valid = dist > 0.5   # skip cells right at the centroid
    if valid.sum() < 3:
        return 0.0

    dx, dy, dist = dx[valid], dy[valid], dist[valid]
    rows_v, cols_v = rows[valid], cols[valid]

    # Unit radial vector (pointing away from center)
    rx = dx / dist
    ry = dy / dist

    # Unit tangential vector (90° counter-clockwise from radial)
    tx = -ry
    ty =  rx

    # Velocity at each cell
    uv = u[rows_v, cols_v]
    vv = v[rows_v, cols_v]

    # Project onto tangential and radial directions
    tang = np.abs(uv * tx + vv * ty)
    rad  = np.abs(uv * rx + vv * ry)

    denom = tang + rad
    if denom.sum() < 1e-10:
        return 0.0

    return float(tang.sum() / denom.sum())


def detect_eddies(
    u: np.ndarray,
    v: np.ndarray,
    land_mask: np.ndarray,
    ow_threshold: float = 2.0,
    min_cells: int = 16,
    circ_threshold: float = 0.55,
) -> list[dict]:
    """
    Detect eddies in a single 2-D velocity field.

    Two-stage filter:
      1. Okubo-Weiss: W < −ow_threshold * std(W)  (rotation-dominated cells)
      2. Circulation check: tangential / (tangential + radial) > circ_threshold
         This rejects shear flows and boundary jets that pass the OW test.

    Args:
        u, v           : (H, W) velocity components (land cells should be 0 or NaN)
        land_mask      : (H, W) bool, True = land
        ow_threshold   : fraction of std(W) below which a cell is an eddy core candidate
        min_cells      : minimum number of grid cells for a valid eddy
        circ_threshold : minimum circulation ratio (0.55 = slightly more tangential
                         than radial; 0.65 = clearly circular)

    Returns:
        List of dicts, one per detected eddy:
          {
            "center_x": float,        # grid index along H (east–west)
            "center_y": float,        # grid index along W (north–south)
            "n_cells": int,           # number of cells in the eddy core
            "type": str,              # "cyclonic" or "anticyclonic"
            "mean_vorticity": float,
            "mean_ow": float,
            "circ_ratio": float,      # tangential fraction (1.0 = perfect circle)
          }
    """
    ocean = ~land_mask

    # Zero land before computing gradients
    u_clean = np.where(ocean, u, 0.0)
    v_clean = np.where(ocean, v, 0.0)

    W, omega = okubo_weiss(u_clean, v_clean)

    # Threshold: OW << 0 means rotation-dominated
    W_ocean = W[ocean]
    if len(W_ocean) == 0 or W_ocean.std() < 1e-12:
        return []
    std_W = W_ocean.std()
    thresh = -ow_threshold * std_W

    # Binary mask: eddy candidate cells (OW below threshold, ocean only)
    eddy_mask = (W < thresh) & ocean

    if eddy_mask.sum() == 0:
        return []

    # Connected components
    labeled, n_components = label(eddy_mask)

    eddies = []
    for comp_id in range(1, n_components + 1):
        comp = labeled == comp_id
        n_cells = int(comp.sum())
        if n_cells < min_cells:
            continue

        # Centroid
        cy, cx = center_of_mass(comp)   # scipy returns (row, col) = (H, W)

        # Circulation check: reject shear flows and boundary jets
        circ = _circulation_ratio(u_clean, v_clean, cx, cy, comp)
        if circ < circ_threshold:
            continue

        mean_vort = float(omega[comp].mean())
        mean_ow   = float(W[comp].mean())
        eddy_type = "cyclonic" if mean_vort > 0 else "anticyclonic"

        eddies.append({
            "center_x": float(cx),      # axis-0 (H, east–west)
            "center_y": float(cy),      # axis-1 (W, north–south)
            "n_cells": n_cells,
            "type": eddy_type,
            "mean_vorticity": mean_vort,
            "mean_ow": mean_ow,
            "circ_ratio": circ,
        })

    return eddies


# ---------------------------------------------------------------------------
# Dataset scan
# ---------------------------------------------------------------------------

SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def scan_dataset(
    pickle_path: str,
    splits: list[int],
    ow_threshold: float,
    min_cells: int,
    circ_threshold: float,
    verbose: bool,
) -> list[dict]:
    """
    Scan every sample in the specified splits and return a flat list of
    eddy records, each tagged with split name and sample index.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Land mask from first sample of first split
    arr0      = data[0]
    land_mask = np.isnan(arr0[:, :, 0, 0])   # (H, W) bool

    all_records = []

    for split_id in splits:
        arr        = data[split_id]            # (H, W, 2, N)
        split_name = SPLIT_NAMES[split_id]
        N          = arr.shape[3]

        n_with_eddies = 0
        n_eddies_total = 0

        for i in range(N):
            # Extract u, v — NaN at land cells; replace with 0 for computation
            u = np.nan_to_num(arr[:, :, 0, i], nan=0.0)
            v = np.nan_to_num(arr[:, :, 1, i], nan=0.0)

            eddies = detect_eddies(u, v, land_mask, ow_threshold, min_cells, circ_threshold)

            for eddy in eddies:
                all_records.append({
                    "split": split_name,
                    "sample_index": i,
                    **eddy,
                })

            if eddies:
                n_with_eddies += 1
                n_eddies_total += len(eddies)

            if verbose and (i % 500 == 0):
                print(f"  [{split_name}] {i:5d}/{N}  eddies so far: {n_eddies_total}")

        pct = 100 * n_with_eddies / N if N > 0 else 0
        print(f"  {split_name:5s}  samples={N:5d}  "
              f"with_eddies={n_with_eddies:5d} ({pct:.1f}%)  "
              f"total_eddies={n_eddies_total}")

    return all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect eddies in ocean current dataset using Okubo-Weiss parameter."
    )
    p.add_argument("--pickle",       default="Datasets/data.pickle",
                   help="Path to dataset pickle (default: Datasets/data.pickle)")
    p.add_argument("--out",          default="eddies/eddy_catalog.json",
                   help="Output JSON catalog path (default: eddies/eddy_catalog.json)")
    p.add_argument("--split",        type=int, nargs="+", default=[0, 1, 2],
                   choices=[0, 1, 2],
                   help="Which splits to scan: 0=train 1=val 2=test (default: all)")
    p.add_argument("--ow_threshold", type=float, default=2.0,
                   help="OW threshold as fraction of std(W). "
                        "Lower = more sensitive, more false positives. (default: 2.0)")
    p.add_argument("--min_cells",    type=int, default=16,
                   help="Minimum grid cells for a valid eddy (default: 16)")
    p.add_argument("--circ_threshold", type=float, default=0.55,
                   help="Min tangential/(tangential+radial) ratio to accept an eddy. "
                        "Rejects shear flows. (default: 0.55)")
    p.add_argument("--verbose",      action="store_true",
                   help="Print progress every 500 samples")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Dataset : {args.pickle}")
    print(f"Splits  : {[SPLIT_NAMES[s] for s in args.split]}")
    print(f"OW threshold  : {args.ow_threshold} × std(W)")
    print(f"Min cells     : {args.min_cells}")
    print(f"Circ threshold: {args.circ_threshold}")
    print()

    records = scan_dataset(
        pickle_path    = args.pickle,
        splits         = args.split,
        ow_threshold   = args.ow_threshold,
        min_cells      = args.min_cells,
        circ_threshold = args.circ_threshold,
        verbose        = args.verbose,
    )

    # Summary
    print()
    n_cyc  = sum(1 for r in records if r["type"] == "cyclonic")
    n_anti = sum(1 for r in records if r["type"] == "anticyclonic")
    print(f"Total eddies detected : {len(records)}")
    print(f"  Cyclonic            : {n_cyc}")
    print(f"  Anticyclonic        : {n_anti}")

    # Save catalog
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "metadata": {
                "pickle":         args.pickle,
                "ow_threshold":   args.ow_threshold,
                "min_cells":      args.min_cells,
                "circ_threshold": args.circ_threshold,
                "splits":         [SPLIT_NAMES[s] for s in args.split],
                "total_eddies":   len(records),
                "cyclonic":       n_cyc,
                "anticyclonic":   n_anti,
            },
            "eddies": records,
        }, f, indent=2)

    print(f"\nCatalog saved → {args.out}")


if __name__ == "__main__":
    main()
