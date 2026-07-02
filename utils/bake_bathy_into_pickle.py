"""
One-time utility: crop the bathymetry field from full_dataset.mat and bake it
into the chrono pickle so ConditionalOceanDataset can use it as a conditioning
channel.

The full domain is 242×329; our crop is 94×44.  The exact slice was determined
by comparing the land mask from the pickle to the full-domain land mask in
full_dataset.mat — whichever row/col offset makes them match.

Usage (from workspace root):
    python Utils/bake_bathy_into_pickle.py \
        --mat      Datasets/full_datasets/full_dataset.mat \
        --pickle   Datasets/pickles/data_divfree_chrono.pickle \
        --out      Datasets/pickles/data_divfree_chrono.pickle   # overwrite in-place
"""

import argparse
import os
import pickle
import sys

import numpy as np

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)


def load_bathy(mat_path: str) -> np.ndarray:
    """Load the `h` array from full_dataset.mat (HDF5 v7.3 format)."""
    try:
        import h5py
        with h5py.File(mat_path, "r") as f:
            h = np.array(f["h"], dtype=np.float32)
        # HDF5 MATLAB files are transposed relative to scipy convention.
        if h.ndim == 2 and h.shape[0] < h.shape[1]:
            h = h.T
        return h
    except Exception as e:
        raise RuntimeError(f"Failed to load {mat_path}: {e}")


def find_crop(land_full: np.ndarray, land_crop: np.ndarray) -> tuple[int, int]:
    """
    Find the (row0, col0) offset such that
        land_full[row0:row0+H, col0:col0+W] == land_crop
    by sliding a window.  Raises if no match is found.
    """
    H, W = land_crop.shape
    Hf, Wf = land_full.shape
    for r in range(Hf - H + 1):
        for c in range(Wf - W + 1):
            if np.array_equal(land_full[r:r+H, c:c+W], land_crop):
                return r, c
    raise ValueError(
        f"Could not align crop ({H}x{W}) inside full domain ({Hf}x{Wf}).  "
        "Check that the pickle's land_mask matches land from full_dataset.mat."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat",    default="Datasets/full_datasets/full_dataset.mat")
    p.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    p.add_argument("--out",    default=None,
                   help="Output pickle path.  Default: overwrite input.")
    p.add_argument("--row0",   type=int, default=None,
                   help="Manual crop row offset (skip auto-alignment).")
    p.add_argument("--col0",   type=int, default=None,
                   help="Manual crop col offset (skip auto-alignment).")
    args = p.parse_args()

    out_path = args.out or args.pickle
    print(f"Loading {args.mat} ...")
    bathy_full = load_bathy(args.mat)
    print(f"  full domain: {bathy_full.shape}  "
          f"min={bathy_full.min():.1f}  max={bathy_full.max():.1f}")

    print(f"Loading {args.pickle} ...")
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    if "bathy" in data:
        print("  Pickle already contains 'bathy' key — overwriting.")

    land_crop = np.asarray(data["land_mask"], dtype=bool)   # (H, W)
    H, W = land_crop.shape
    print(f"  crop size: {H}x{W}")

    if args.row0 is not None and args.col0 is not None:
        r0, c0 = args.row0, args.col0
        print(f"  Using manual offset: row0={r0}  col0={c0}")
    else:
        print("  Auto-aligning land mask ...")
        # Build binary land mask from full dataset.  We try the `land` key first,
        # then derive it from the first field (zero = land).
        try:
            import h5py
            with h5py.File(args.mat, "r") as f:
                keys = list(f.keys())
            print(f"  full_dataset.mat keys: {keys}")
            with h5py.File(args.mat, "r") as f:
                if "mask" in f:
                    land_full_bool = np.array(f["mask"], dtype=bool).T
                elif "h" in f:
                    # Depth < 0 or 0 typically means land in ocean datasets
                    h_arr = np.array(f["h"], dtype=np.float32)
                    if h_arr.shape[0] < h_arr.shape[1]:
                        h_arr = h_arr.T
                    land_full_bool = (h_arr <= 0)
                else:
                    raise KeyError("No 'mask' or 'h' key in full_dataset.mat")
        except Exception as e:
            raise RuntimeError(f"Cannot build full land mask: {e}")
        print(f"  full land mask: {land_full_bool.shape}")
        r0, c0 = find_crop(land_full_bool, land_crop)
        print(f"  Found alignment: row0={r0}  col0={c0}")

    bathy_crop = bathy_full[r0:r0+H, c0:c0+W].astype(np.float32)
    # Set land cells to 0 (will also be zeroed by geometry_channels normalization).
    bathy_crop[land_crop] = 0.0
    print(f"  bathy crop: min={bathy_crop[~land_crop].min():.1f}  "
          f"max={bathy_crop[~land_crop].max():.1f}  "
          f"mean={bathy_crop[~land_crop].mean():.1f}")

    data["bathy"] = bathy_crop
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved to {out_path}  (bathy shape={bathy_crop.shape})")


if __name__ == "__main__":
    main()
