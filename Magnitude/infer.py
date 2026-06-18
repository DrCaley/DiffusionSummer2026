"""
Batch inference + visualization for the UNet speed (magnitude) regressor.

Mirrors GP Baseline/batch_magnitude.py so the two approaches can be compared
directly on the same validation samples / seeds.  For each run it samples a
biased robot path, predicts the dense speed field, reports speed RMSE / MAE /
relRMSE, and saves a 1×3 plot (GT speed + path | predicted speed | error).

Usage:
    python Magnitude/infer.py --pickle Datasets/data.pickle \
        --checkpoint Magnitude/checkpoints/best_magnitude_unet.pt \
        --n_runs 10 --random --seed 1234 --path_steps 150
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
sys.path.insert(0, _here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "utils"))

from dataset import OceanCurrentDataset
from paths   import biased_walk_path
from model   import MagnitudeUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="Datasets/data.pickle")
    p.add_argument("--checkpoint", default="Magnitude/checkpoints/best_magnitude_unet.pt")
    p.add_argument("--n_runs",     type=int, default=10)
    p.add_argument("--random",     action="store_true")
    p.add_argument("--seed",       type=int, default=1234)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--device",     default=None)
    p.add_argument("--out_dir",    default="Magnitude/results")
    return p.parse_args()


def pick_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, spd, path_mask, land_mask, speed_mean, speed_std, device):
    """Run the UNet on one sample; returns dense speed field (H, W) in phys units."""
    obs_speed = np.zeros_like(spd)
    obs_speed[path_mask] = spd[path_mask] / speed_std
    inp = np.stack([
        obs_speed,
        path_mask.astype(np.float32),
        land_mask.astype(np.float32),
    ], axis=0)[None]                                  # (1, 3, H, W)
    inp_t = torch.from_numpy(inp).to(device)
    pred = model(inp_t)[0, 0].cpu().numpy()           # (H, W) standardized
    pred = pred * speed_std + speed_mean              # un-standardize
    pred = np.clip(pred, 0.0, None)
    pred[land_mask] = 0.0
    return pred.astype(np.float32)


def save_plot(mag_true, mag_pred, err, land_mask, path_mask,
              rmse, path_cells, label, sample_idx, seed, out_path):
    mag_true_d, mag_pred_d, err_d = mag_true.T, mag_pred.T, err.T
    land_d, path_d = land_mask.T, path_mask.T

    vmax = float(np.nanmax(np.ma.masked_where(land_d, mag_true_d)))
    extent = [-0.5, land_d.shape[1] - 0.5, -0.5, land_d.shape[0] - 0.5]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    gt = np.ma.masked_where(land_d, mag_true_d)
    im0 = axes[0].imshow(gt, origin="lower", cmap="viridis", aspect="auto",
                         extent=extent, vmin=0, vmax=vmax)
    path_display = np.ma.masked_where(~path_d, np.ones_like(land_d, dtype=float))
    axes[0].imshow(path_display, origin="lower", cmap="Reds", alpha=0.9,
                   aspect="auto", extent=extent, vmin=0, vmax=1, zorder=2)
    axes[0].imshow(land_d, origin="lower",
                   cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
                   aspect="auto", extent=extent, zorder=1)
    plt.colorbar(im0, ax=axes[0], label="speed |v|", shrink=0.8)
    axes[0].set_title(f"Ground-truth speed  (path = {path_cells} cells)")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")

    pr = np.ma.masked_where(land_d, mag_pred_d)
    im1 = axes[1].imshow(pr, origin="lower", cmap="viridis", aspect="auto",
                         extent=extent, vmin=0, vmax=vmax)
    axes[1].imshow(land_d, origin="lower",
                   cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
                   aspect="auto", extent=extent, zorder=1)
    plt.colorbar(im1, ax=axes[1], label="speed |v|", shrink=0.8)
    axes[1].set_title("Predicted speed (UNet)")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")

    er = np.ma.masked_where(land_d, err_d)
    im2 = axes[2].imshow(er, origin="lower", cmap="hot_r", aspect="auto", extent=extent)
    axes[2].imshow(land_d, origin="lower",
                   cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
                   aspect="auto", extent=extent, zorder=1)
    plt.colorbar(im2, ax=axes[2], label="|error|", shrink=0.8)
    axes[2].set_title(f"Speed error  (RMSE = {rmse:.4f})")
    axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")

    plt.suptitle(f"Run {label} — Val sample {sample_idx}, path seed {seed}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device(args.device)
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    speed_mean = ckpt["speed_mean"]
    speed_std  = ckpt["speed_std"]
    base_ch    = ckpt.get("args", {}).get("base_ch", 64)
    model = MagnitudeUNet(in_ch=3, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.5f})")

    val_ds    = OceanCurrentDataset(args.pickle, split=1)
    land_mask = val_ds.land_mask.numpy().astype(bool)
    n_ocean   = int((~land_mask).sum())

    if args.random:
        rng         = np.random.default_rng(args.seed)
        sample_idxs = rng.integers(0, len(val_ds), size=args.n_runs)
        seeds       = rng.integers(0, 1_000_000, size=args.n_runs)
    else:
        sample_idxs = [i % len(val_ds) for i in range(args.n_runs)]
        seeds       = [1000 + i for i in range(args.n_runs)]

    rmse_list, mae_list, rel_list = [], [], []

    for i in range(args.n_runs):
        sample_idx = int(sample_idxs[i])
        seed       = int(seeds[i])

        x0  = val_ds[sample_idx].numpy()
        spd = np.sqrt(x0[0] ** 2 + x0[1] ** 2).astype(np.float32)
        spd[land_mask] = 0.0

        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps, seed=seed)
        path_mask &= ~land_mask

        mag_pred = predict(model, spd, path_mask, land_mask, speed_mean, speed_std, device)

        err = np.abs(mag_pred - spd)
        err[land_mask] = np.nan
        ocean_err = err[~land_mask]
        rmse = float(np.sqrt(np.nanmean(ocean_err ** 2)))
        mae  = float(np.nanmean(ocean_err))
        mean_speed = float(np.nanmean(spd[~land_mask]))
        rel_rmse   = rmse / (mean_speed + 1e-8)

        path_cells = int(path_mask.sum())
        pct = 100 * path_cells / n_ocean
        print(f"\nRun {i+1}/{args.n_runs}  (val sample {sample_idx}, seed {seed})")
        print(f"  Path cells: {path_cells} ({pct:.1f}%)   "
              f"RMSE: {rmse:.4f}   MAE: {mae:.4f}   relRMSE: {rel_rmse:.3f}")
        rmse_list.append(rmse); mae_list.append(mae); rel_list.append(rel_rmse)

        label    = f"{i+1:02d}"
        out_path = os.path.join(args.out_dir, f"mag_val{sample_idx}_{label}.png")
        save_plot(spd, mag_pred, err, land_mask, path_mask,
                  rmse, path_cells, label, sample_idx, seed, out_path)
        print(f"  Saved: {out_path}")

    print(f"\n{'=' * 56}")
    print(f"Magnitude UNet over {args.n_runs} runs")
    print(f"  Mean RMSE:    {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"  Mean MAE:     {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"  Mean relRMSE: {np.mean(rel_list):.3f} ± {np.std(rel_list):.3f}")
    print(f"  RMSE min/max: {np.min(rmse_list):.4f} / {np.max(rmse_list):.4f}")


if __name__ == "__main__":
    main()
