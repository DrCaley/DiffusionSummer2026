"""
build_tidal_pickle.py
=====================
Convert stjohn_hourly_5m_velocity_ramhead_v2.mat to data_tidal.pickle
with precomputed M2 + S2 tidal phase features per timestep.

Features (4-dim):
    [sin(M2_phase), cos(M2_phase), sin(S2_phase), cos(S2_phase)]

Tidal periods (in days, matching ocean_time units):
    M2 = 12.4206 h / 24 = 0.517525 days
    S2 = 12.0000 h / 24 = 0.500000 days

Pickle format (dict):
    "train" / "val" / "test" : (H, W, 2, N) float32
    "features" : {"train": (N, 4), "val": (N, 4), "test": (N, 4)} float32
    "feature_names" : ["sin_M2", "cos_M2", "sin_S2", "cos_S2"]
    "start_indices" : {"train": 0, "val": n_train, "test": n_train+n_val}

Usage:
    python build_tidal_pickle.py \
        --mat stjohn_hourly_5m_velocity_ramhead_v2.mat \
        --out data_tidal.pickle
"""

import argparse
import os
import pickle
import numpy as np
from scipy.io import loadmat

M2_PERIOD_DAYS = 12.4206 / 24.0   # 0.517525 days
S2_PERIOD_DAYS = 12.0000 / 24.0   # 0.500000 days


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="stjohn_hourly_5m_velocity_ramhead_v2.mat")
    p.add_argument("--out", default="data_tidal.pickle")
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac",   type=float, default=0.15)
    return p.parse_args()


def tidal_features(ocean_time: np.ndarray) -> np.ndarray:
    """
    Compute 4 tidal phase features for each timestep.
    ocean_time: (N,) float64 in MATLAB datenum days.
    Returns: (N, 4) float32 [sin_M2, cos_M2, sin_S2, cos_S2]
    """
    phase_M2 = (ocean_time % M2_PERIOD_DAYS) / M2_PERIOD_DAYS * 2.0 * np.pi
    phase_S2 = (ocean_time % S2_PERIOD_DAYS) / S2_PERIOD_DAYS * 2.0 * np.pi
    feats = np.stack([
        np.sin(phase_M2),
        np.cos(phase_M2),
        np.sin(phase_S2),
        np.cos(phase_S2),
    ], axis=1).astype(np.float32)  # (N, 4)
    return feats


def main():
    args = parse_args()

    print(f"Loading {args.mat} ...")
    m = loadmat(args.mat)
    u = m["u"].astype(np.float32)          # (H, W, T)
    v = m["v"].astype(np.float32)
    ocean_time = m["ocean_time"].flatten()  # (T,) float64 days
    H, W, T = u.shape

    print(f"  Grid : {H}×{W}, T={T} timesteps")
    print(f"  ocean_time[0]={ocean_time[0]:.6f}, diff={ocean_time[1]-ocean_time[0]:.6f} days")

    # Chronological split
    n_train = int(T * args.train_frac)
    n_val   = int(T * args.val_frac)
    n_test  = T - n_train - n_val
    i_val   = n_train
    i_test  = n_train + n_val
    print(f"  Split: train={n_train}, val={n_val}, test={n_test}")

    fields = np.stack([u, v], axis=2)  # (H, W, 2, T)

    splits = {
        "train": fields[:, :, :, :n_train].copy(),
        "val":   fields[:, :, :, i_val:i_test].copy(),
        "test":  fields[:, :, :, i_test:].copy(),
    }

    # Precompute tidal features for each split
    all_feats = tidal_features(ocean_time)  # (T, 4)
    features = {
        "train": all_feats[:n_train],
        "val":   all_feats[i_val:i_test],
        "test":  all_feats[i_test:],
    }

    start_indices = {"train": 0, "val": n_train, "test": i_test}

    payload = {
        "train":         splits["train"],
        "val":           splits["val"],
        "test":          splits["test"],
        "features":      features,
        "feature_names": ["sin_M2", "cos_M2", "sin_S2", "cos_S2"],
        "start_indices": start_indices,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nSaved {args.out}  ({size_mb:.1f} MB)")
    for k in ("train", "val", "test"):
        print(f"  {k}: velocity={splits[k].shape}, features={features[k].shape}")
    print(f"  feature_names: {payload['feature_names']}")

    # Sanity: show M2 phase range
    f_train = features["train"]
    print(f"\nFeature stats (train):")
    for i, name in enumerate(payload["feature_names"]):
        print(f"  {name}: min={f_train[:,i].min():.3f}, max={f_train[:,i].max():.3f}")


if __name__ == "__main__":
    main()
