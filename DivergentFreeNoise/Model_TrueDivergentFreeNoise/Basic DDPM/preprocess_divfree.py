from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _load_pickle_splits(pickle_path: Path) -> list[np.ndarray]:
    with open(pickle_path, "rb") as handle:
        splits = pickle.load(handle)
    if not isinstance(splits, list) or len(splits) != 3:
        raise ValueError("Expected a pickle list with train/val/test arrays")
    return splits


def _save_pickle_splits(splits: list[np.ndarray], pickle_path: Path) -> None:
    with open(pickle_path, "wb") as handle:
        pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _project_divergence_free_numpy(field: np.ndarray) -> np.ndarray:
    if field.ndim != 3 or field.shape[0] != 2:
        raise ValueError(f"Expected field shape (2, H, W), got {field.shape}")

    _, height, width = field.shape
    ky, kx = np.meshgrid(2.0 * np.pi * np.fft.fftfreq(height), 2.0 * np.pi * np.fft.fftfreq(width), indexing="ij")
    u_hat = np.fft.fft2(field[0])
    v_hat = np.fft.fft2(field[1])
    dot = kx * u_hat + ky * v_hat
    k2 = kx**2 + ky**2

    factor = np.zeros_like(dot)
    valid = k2 > 0.0
    factor[valid] = dot[valid] / k2[valid]

    u_hat = u_hat - kx * factor
    v_hat = v_hat - ky * factor
    projected = np.stack([np.fft.ifft2(u_hat).real, np.fft.ifft2(v_hat).real], axis=0)
    return projected.astype(np.float32, copy=False)


def _build_land_mask(split: np.ndarray) -> np.ndarray:
    if split.ndim != 4 or split.shape[2] != 2:
        raise ValueError(f"Expected split shape (H, W, 2, N), got {split.shape}")
    return np.isnan(split).any(axis=(2, 3))


def _to_nchw(split: np.ndarray) -> np.ndarray:
    return np.nan_to_num(split, nan=0.0).transpose(3, 2, 0, 1).astype(np.float32, copy=False)


def _project_divergence_free_batch(batch: np.ndarray) -> np.ndarray:
    if batch.ndim != 4 or batch.shape[1] != 2:
        raise ValueError(f"Expected batch shape (N, 2, H, W), got {batch.shape}")

    _, _, height, width = batch.shape
    ky, kx = np.meshgrid(2.0 * np.pi * np.fft.fftfreq(height), 2.0 * np.pi * np.fft.fftfreq(width), indexing="ij")
    ky = ky[None, :, :]
    kx = kx[None, :, :]

    u_hat = np.fft.fft2(batch[:, 0], axes=(-2, -1))
    v_hat = np.fft.fft2(batch[:, 1], axes=(-2, -1))
    dot = kx * u_hat + ky * v_hat
    k2 = kx**2 + ky**2

    factor = np.zeros_like(dot)
    np.divide(dot, k2, out=factor, where=k2 > 0.0)

    u_hat = u_hat - kx * factor
    v_hat = v_hat - ky * factor
    projected = np.stack([np.fft.ifft2(u_hat, axes=(-2, -1)).real, np.fft.ifft2(v_hat, axes=(-2, -1)).real], axis=1)
    return projected.astype(np.float32, copy=False)


def prepare_divergence_free_pickle(source_pickle: Path, target_pickle: Path) -> np.ndarray:
    splits = _load_pickle_splits(source_pickle)
    land_mask = _build_land_mask(splits[0])

    cleaned_splits: list[np.ndarray] = []
    for split in splits:
        nchw = _to_nchw(split)
        projected_nchw = _project_divergence_free_batch(nchw)
        projected_nchw[:, :, land_mask] = np.nan
        cleaned_splits.append(projected_nchw.transpose(2, 3, 1, 0).astype(np.float32, copy=False))

    _save_pickle_splits(cleaned_splits, target_pickle)
    return land_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a divergence-free pickle while preserving the land mask.")
    parser.add_argument("--source", type=Path, default=Path("..") / "data.pickle")
    parser.add_argument("--target", type=Path, default=Path("..") / "data_divfree.pickle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    prepare_divergence_free_pickle(args.source, args.target)
    print(f"Saved divergence-free pickle to {args.target}")


if __name__ == "__main__":
    main()
