"""
One-time utility: extract bathymetry from full_dataset.mat and bake it into
the chrono pickle so ConditionalOceanDataset can use it as an optional 4th
geometry channel (enabled automatically when the pickle contains 'bathy').

Alignment is done by nearest-neighbour matching of lat/lon coordinates from
ramhead_dataset.mat against the full domain lat_rho/lon_rho grid, so the
crop is exact regardless of the grid projection.

Usage (from workspace root):
    python Utils/bake_bathy_into_pickle.py
    python Utils/bake_bathy_into_pickle.py --verify   # show alignment stats, don't save
"""

import argparse
import os
import pickle
import sys

import numpy as np
import scipy.io as sio

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)


def load_full_dataset(mat_path: str):
    """Load h, lat_rho, lon_rho, mask from full_dataset.mat (HDF5 v7.3)."""
    import h5py
    with h5py.File(mat_path, "r") as f:
        h        = np.array(f["h"][()],       dtype=np.float64)  # (242, 329)
        lat_rho  = np.array(f["lat_rho"][()], dtype=np.float64)
        lon_rho  = np.array(f["lon_rho"][()], dtype=np.float64)
        mask     = np.array(f["mask"][()],    dtype=np.float64)  # 1=ocean 0=land
    return h, lat_rho, lon_rho, mask


def load_ramhead_coords(mat_path: str):
    """Load lat/lon (94×44) from ramhead_dataset.mat (old scipy format)."""
    m = sio.loadmat(mat_path)
    return m["lat"].astype(np.float64), m["lon"].astype(np.float64)


def align_bathy(h_full, lat_full, lon_full, lat_crop, lon_crop):
    """
    For each cell (i,j) in the 94×44 crop grid, find the nearest cell in the
    full (242×329) grid by Euclidean distance in (lat, lon) space and read h.

    Returns bathy_crop (94, 44) float32 with the matched depth values.
    Also returns mean_dist_km for alignment quality reporting.
    """
    H, W = lat_crop.shape
    Hf, Wf = lat_full.shape

    # Flatten full grid for fast nearest-neighbour lookup.
    lat_f = lat_full.ravel()
    lon_f = lon_full.ravel()
    h_f   = h_full.ravel()

    bathy_crop = np.zeros((H, W), dtype=np.float32)
    dists = []

    for i in range(H):
        dlat = lat_f - lat_crop[i, :][:, None]     # (W, Nfull)
        dlon = lon_f - lon_crop[i, :][:, None]
        # Approximate km distance (1 deg lat ≈ 111 km; lon scaled by cos(lat)).
        cos_lat = np.cos(np.radians(lat_crop[i, :]))[:, None]
        d2 = (dlat * 111.0) ** 2 + (dlon * 111.0 * cos_lat) ** 2
        nn = np.argmin(d2, axis=1)                  # (W,)
        bathy_crop[i, :] = h_f[nn].astype(np.float32)
        dists.extend(np.sqrt(d2[np.arange(W), nn]).tolist())

    mean_dist_km = float(np.mean(dists))
    max_dist_km  = float(np.max(dists))
    return bathy_crop, mean_dist_km, max_dist_km


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full_mat",    default="Datasets/full_datasets/full_dataset.mat")
    p.add_argument("--ramhead_mat", default="Datasets/full_datasets/ramhead_dataset.mat")
    p.add_argument("--pickle",      default="Datasets/pickles/data_divfree_chrono.pickle")
    p.add_argument("--out",         default=None,
                   help="Output pickle path. Default: overwrite input in-place.")
    p.add_argument("--verify",      action="store_true",
                   help="Print alignment stats and exit without saving.")
    args = p.parse_args()

    out_path = args.out or args.pickle

    print(f"Loading full domain from {args.full_mat} ...")
    h_full, lat_full, lon_full, mask_full = load_full_dataset(args.full_mat)
    print(f"  h: {h_full.shape}  lat: {lat_full.shape}  "
          f"depth range [{h_full.min():.1f}, {h_full.max():.1f}] m")

    print(f"Loading crop coordinates from {args.ramhead_mat} ...")
    lat_crop, lon_crop = load_ramhead_coords(args.ramhead_mat)
    print(f"  lat: {lat_crop.shape}  "
          f"lat range [{lat_crop.min():.4f}, {lat_crop.max():.4f}]  "
          f"lon range [{lon_crop.min():.4f}, {lon_crop.max():.4f}]")

    print("Aligning bathymetry via nearest-neighbour lat/lon matching ...")
    bathy_crop, mean_km, max_km = align_bathy(
        h_full, lat_full, lon_full, lat_crop, lon_crop)
    print(f"  Match quality: mean={mean_km:.4f} km  max={max_km:.4f} km")
    if max_km > 0.5:
        print(f"  WARNING: max distance {max_km:.3f} km is large — "
              "check that both .mat files cover the same domain.")

    print(f"Loading {args.pickle} ...")
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    land_crop = np.asarray(data["land_mask"], dtype=bool)

    # Zero land cells (geometry_channels will also zero them, but be explicit).
    bathy_crop[land_crop] = 0.0
    ocean_crop = ~land_crop
    print(f"  bathy ocean cells: min={bathy_crop[ocean_crop].min():.1f}  "
          f"max={bathy_crop[ocean_crop].max():.1f}  "
          f"mean={bathy_crop[ocean_crop].mean():.1f} m")

    # Cross-check: land in full_dataset mask vs our land_mask.
    ocean_full = (mask_full > 0.5)
    # Use nearest-neighbour to pull the full-domain ocean flag at each crop cell.
    H, W = lat_crop.shape
    Hf, Wf = lat_full.shape
    lat_f = lat_full.ravel(); lon_f = lon_full.ravel()
    mask_f = ocean_full.ravel().astype(float)
    ocean_nn = np.zeros((H, W), dtype=bool)
    for i in range(H):
        dlat = lat_f - lat_crop[i, :][:, None]
        dlon = lon_f - lon_crop[i, :][:, None]
        cos_lat = np.cos(np.radians(lat_crop[i, :]))[:, None]
        d2 = (dlat * 111.0) ** 2 + (dlon * 111.0 * cos_lat) ** 2
        nn = np.argmin(d2, axis=1)
        ocean_nn[i, :] = mask_f[nn] > 0.5
    land_nn = ~ocean_nn
    agreement = float((land_nn == land_crop).mean())
    print(f"  Land-mask agreement with full_dataset: {agreement:.1%}")
    if agreement < 0.97:
        print("  WARNING: land mask agreement < 97% — verify domain coverage.")

    if args.verify:
        print("--verify mode: not saving.")
        return

    if "bathy" in data:
        print("  Overwriting existing 'bathy' key in pickle.")
    data["bathy"] = bathy_crop
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved to {out_path}  (bathy shape={bathy_crop.shape})")
    print("To activate: re-run training — ConditionalOceanDataset detects 'bathy' automatically.")


if __name__ == "__main__":
    main()
