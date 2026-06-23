"""
Compute a 2D spectral amplitude filter from the training data.

The filter is sqrt(P_avg(kx, ky)) where P_avg is the mean power spectrum
across all training samples and both channels, with the ocean mean subtracted
per sample (so DC doesn't dominate).

When this filter is applied in Fourier space during noise generation, the noise
has the same spectral shape as the data -- energy concentrated at large scales,
matching the dominant low-wavenumber ocean current structure. This gives the
diffusion model a uniform signal-to-noise ratio across wavenumbers instead of
trivially easy high-k modes and very hard low-k modes.

Usage (from workspace root):
    python utils/compute_spectral_filter.py \
        --pickle Datasets/data.pickle \
        --out    Datasets/spectral_filter.npy
"""

import argparse
import pickle

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle", default="Datasets/data.pickle")
    p.add_argument("--out",    default="Datasets/spectral_filter.npy")
    p.add_argument("--n",      type=int, default=None,
                   help="Max training samples (default: all)")
    args = p.parse_args()

    print(f"Loading {args.pickle} ...")
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    arr       = data[0]               # training split: (H, W, 2, N)
    land_mask = np.isnan(arr[:, :, 0, 0])
    H, W, C, N = arr.shape
    n = min(N, args.n) if args.n else N

    print(f"Grid: {H}x{W}   channels: {C}   samples used: {n}")

    P_sum = np.zeros((H, W), dtype=np.float64)
    count = 0
    for i in range(n):
        if i % 1000 == 0:
            print(f"  {i}/{n} ...", flush=True)
        uv = np.nan_to_num(arr[:, :, :, i], nan=0.0)   # (H, W, 2)
        for c in range(C):
            field = uv[:, :, c]
            # Remove per-sample ocean mean so DC doesn't contaminate the filter
            ocean_mean = field[~land_mask].mean()
            field = field - ocean_mean
            P_sum += np.abs(np.fft.fft2(field)) ** 2
            count += 1

    P_avg = P_sum / count               # (H, W) average power spectrum

    # Zero DC — we don't want to amplify the mean-flow component
    P_avg[0, 0] = 0.0

    # Amplitude filter = sqrt(P_avg)
    amp_filter = np.sqrt(P_avg).astype(np.float32)

    # Normalize: mean over nonzero modes = 1.0
    # The final per-sample std normalisation in div_free_noise handles absolute scale.
    nonzero = amp_filter > 0
    amp_filter = amp_filter / amp_filter[nonzero].mean()

    np.save(args.out, amp_filter)

    print(f"\nSaved: {args.out}  shape={amp_filter.shape}")
    print(f"  min={amp_filter.min():.4f}  max={amp_filter.max():.4f}  "
          f"mean(nonzero)={amp_filter[nonzero].mean():.4f}")


if __name__ == "__main__":
    main()
