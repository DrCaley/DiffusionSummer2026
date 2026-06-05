"""
Run RePaint inference 10 times on the validation set with different random robot paths
and save individual 2x2 plots labelled 1-10.

Usage:
    python3 batch_infer.py
    python3 batch_infer.py --n_runs 10 --path_steps 300 --resample 10
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from model         import UNet
from plot_utils    import plot_field
from repaint_infer import biased_walk_path, repaint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--checkpoint",  default="checkpoints/best_model.pt")
    p.add_argument("--n_runs",      type=int, default=10,  help="number of runs")
    p.add_argument("--path_steps",  type=int, default=150, help="robot walk length")
    p.add_argument("--resample",    type=int, default=10,  help="RePaint r parameter")
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--time_dim",    type=int, default=256)
    p.add_argument("--out_dir",     default="batch_results")
    return p.parse_args()


def run_one(model, diffusion, val_ds, land_mask_np, sample_idx, seed, args, device):
    x0_true = val_ds[sample_idx]
    u_true  = x0_true[0].numpy()
    v_true  = x0_true[1].numpy()

    path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

    x0_observed = x0_true.clone()
    path_t = torch.from_numpy(path_mask)
    x0_observed[:, ~path_t] = 0.0

    x0_pred = repaint(
        model, diffusion, x0_observed,
        path_mask, land_mask_np,
        r=args.resample, device=device,
    )
    u_pred = x0_pred[0].numpy()
    v_pred = x0_pred[1].numpy()

    err = np.sqrt((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)
    err[land_mask_np] = np.nan
    ocean_err = err[~land_mask_np]
    rmse = float(np.sqrt(np.nanmean(ocean_err ** 2)))

    # Transpose for display: (94,44) -> (44,94), X=0-93, Y=0-43
    u_true_d    = u_true.T
    v_true_d    = v_true.T
    u_pred_d    = u_pred.T
    v_pred_d    = v_pred.T
    land_d      = land_mask_np.T
    path_d      = path_mask.T
    err_d       = err.T

    return u_true_d, v_true_d, u_pred_d, v_pred_d, land_d, path_d, err_d, rmse, path_mask.sum()


def save_plot(u_true, v_true, u_pred, v_pred, land_mask, path_mask, err, rmse,
              path_cells, label, sample_idx, seed, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Ground truth
    plot_field(axes[0], u_true, v_true, land_mask, "Ground Truth")

    # 2. Robot path
    axes[1].imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=0,
    )
    path_display = np.zeros_like(land_mask, dtype=float)
    path_display[path_mask] = 1.0
    axes[1].imshow(
        path_display, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    axes[1].set_title(f"Robot Path ({path_cells} cells, seed={seed})", fontsize=11)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728",                 label="Path")
    land_p  = mpatches.Patch(facecolor="black",                   label="Land")
    axes[1].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

    # 3. Reconstruction
    plot_field(axes[2], u_pred, v_pred, land_mask, "Reconstructed (RePaint)", cmap="cool")

    # 4. Error map
    err_plot = np.ma.masked_where(land_mask, err)
    im = axes[3].imshow(
        err_plot, origin="lower", cmap="hot_r", aspect="auto",
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
    )
    axes[3].imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
        extent=[-0.5, land_mask.shape[1] - 0.5, -0.5, land_mask.shape[0] - 0.5],
        aspect="auto", zorder=1,
    )
    plt.colorbar(im, ax=axes[3], label="|error| speed", shrink=0.7)
    axes[3].set_title(f"Error  (RMSE={rmse:.4f})", fontsize=11)
    axes[3].set_xlabel("X"); axes[3].set_ylabel("Y")

    plt.suptitle(
        f"Run {label}  —  Val sample {sample_idx}, path seed {seed}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}  (RMSE={rmse:.4f})")


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    val_ds    = OceanCurrentDataset(args.pickle, split=1)
    land_mask = val_ds.land_mask.numpy()

    # Load model
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)

    model = UNet(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}), T={T}")

    diffusion = DDPM(T=T, beta_schedule="cosine", device=device)

    # 10 runs: val samples 0-9, seeds 0-9
    rmse_list = []
    for i in range(args.n_runs):
        label      = i + 1
        sample_idx = i          # val sample index
        seed       = i * 7 + 1  # distinct seeds spread apart

        print(f"\n[Run {label}/10]  val sample={sample_idx}, seed={seed}")
        (u_true, v_true, u_pred, v_pred,
         land_d, path_d, err_d, rmse, path_cells) = run_one(
            model, diffusion, val_ds, land_mask,
            sample_idx, seed, args, device,
        )
        rmse_list.append(rmse)
        out_path = os.path.join(args.out_dir, f"result_{label:02d}.png")
        save_plot(u_true, v_true, u_pred, v_pred, land_d, path_d, err_d,
                  rmse, path_cells, label, sample_idx, seed, out_path)

    print(f"\nAll done.")
    print(f"RMSE per run: {[f'{r:.4f}' for r in rmse_list]}")
    print(f"Mean RMSE: {np.mean(rmse_list):.4f}  Std: {np.std(rmse_list):.4f}")


if __name__ == "__main__":
    main()
