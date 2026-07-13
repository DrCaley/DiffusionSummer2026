"""
infer_autoencoder_visualize.py
================================
Plain encode-decode inference with the FNO autoencoder (no diffusion, no
RePaint) — pass clean test fields through encoder+decoder and visualize
ground truth vs reconstruction vs error, same quiver/error-map style as
infer_batch_3methods.py.

Usage:
    python3 infer_autoencoder_visualize.py \\
        --pickle /root/ocean_ddpm/data_local.pickle \\
        --checkpoint /root/NeuralOperator/checkpoints_fno_autoencoder/best_fno_autoencoder.pt \\
        --out_dir /root/NeuralOperator/results/autoencoder_visualize \\
        --n_samples 5
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


def _find_diffusion_dir(explicit=None):
    candidates = [explicit] if explicit else []
    candidates += [
        os.path.join(_SCRIPT_DIR, "..", "Repaint vs DPS"),
        os.path.join(_SCRIPT_DIR, "..", "Repaint_vs_DPS"),
        "/root/Repaint_vs_DPS",
    ]
    for d in candidates:
        if not d:
            continue
        d = os.path.abspath(d)
        if os.path.isfile(os.path.join(d, "dataset.py")):
            return d
    raise RuntimeError(f"Cannot find dataset.py — tried: {candidates}")


def _ext(H, W):
    return [-0.5, W - 0.5, -0.5, H - 0.5]


def plot_quiver(ax, field, land_mask, vmax_spd, title, step=2):
    H, W  = land_mask.shape
    u, v  = field[0].T, field[1].T
    lm    = land_mask.T
    speed = np.ma.masked_where(lm, np.sqrt(u**2 + v**2))

    im = ax.imshow(speed, origin="lower", cmap="cool", vmin=0, vmax=vmax_spd,
                   extent=_ext(W, H), aspect="auto", zorder=0)
    ax.imshow(lm, origin="lower",
              cmap=mcolors.ListedColormap(["none", "black"]),
              extent=_ext(W, H), aspect="auto", zorder=1)

    yq, xq = np.mgrid[0:W:step, 0:H:step]
    mask = ~lm[::step, ::step]
    ax.quiver(xq[mask], yq[mask], u[::step, ::step][mask], v[::step, ::step][mask],
              color="black", scale=12, width=0.003, zorder=2)

    ax.set_xlim(-0.5, H - 0.5); ax.set_ylim(-0.5, W - 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(title, fontsize=9)
    return im


def plot_error(ax, pred, truth, land_mask, vmax_err, title):
    H, W = land_mask.shape
    err  = np.sqrt((pred[0]-truth[0])**2 + (pred[1]-truth[1])**2)
    ed   = err.T.astype(float)
    ed[land_mask.T] = np.nan

    ax.imshow(np.zeros((W, H, 3)), origin="lower", extent=_ext(W, H), aspect="auto")
    im = ax.imshow(ed, origin="lower", cmap="hot_r",
                   norm=mcolors.Normalize(0, vmax_err),
                   extent=_ext(W, H), aspect="auto")

    ax.set_xlim(-0.5, H - 0.5); ax.set_ylim(-0.5, W - 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(title, fontsize=9)
    return im


def compute_errors(pred, truth, ocean_mask, eps=1e-8):
    pu, pv = pred[0][ocean_mask],  pred[1][ocean_mask]
    tu, tv = truth[0][ocean_mask], truth[1][ocean_mask]
    rmse = float(np.sqrt(np.mean((pu - tu)**2 + (pv - tv)**2)))
    pm, tm = np.sqrt(pu**2 + pv**2), np.sqrt(tu**2 + tv**2)
    mag_err = float(np.mean(np.abs(pm - tm)))
    cos = np.clip((pu*tu + pv*tv) / (pm*tm + eps), -1, 1)
    ang_err = float(np.degrees(np.mean(np.arccos(cos))))
    return rmse, mag_err, ang_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",        default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--diffusion_dir", default=None)
    p.add_argument("--out_dir",       default="autoencoder_visualize")
    p.add_argument("--n_samples",     type=int, default=5)
    args = p.parse_args()

    diff_dir = _find_diffusion_dir(args.diffusion_dir)
    sys.path.insert(0, diff_dir)
    print(f"Using dataset helpers from: {diff_dir}")

    from dataset import OceanCurrentDataset
    from model_fno_autoencoder import FNOAutoencoder

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ae_args = ckpt.get("args", {})
    model = FNOAutoencoder(
        in_ch=2, base=ae_args.get("base", 32), latent_ch=ae_args.get("latent_ch", 8),
        modes1=ae_args.get("modes1", 12), modes2=ae_args.get("modes2", 6),
        n_blocks=ae_args.get("n_blocks", 2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f})")

    test_ds    = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = test_ds.land_mask.numpy()
    ocean_mask = ~land_mask
    n_test     = len(test_ds)

    sample_indices = list(range(0, n_test, max(1, n_test // args.n_samples)))[:args.n_samples]
    print(f"Test set size: {n_test}. Sample indices: {sample_indices}\n")

    results = []
    with torch.no_grad():
        for idx in sample_indices:
            x0 = test_ds[idx]
            true_np = x0.numpy()
            pred_np = model(x0.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

            rmse, mag_err, ang_err = compute_errors(pred_np, true_np, ocean_mask)
            results.append((idx, rmse, mag_err, ang_err))
            print(f"  idx={idx:5d}  rmse={rmse:.5f}  mag_err={mag_err:.5f}  angle_err={ang_err:.3f}°")

            spd_true = np.sqrt(true_np[0]**2 + true_np[1]**2)
            vmax_spd = float(np.percentile(spd_true[ocean_mask], 98))
            err = np.sqrt((pred_np[0]-true_np[0])**2 + (pred_np[1]-true_np[1])**2)
            vmax_err = float(np.percentile(err[ocean_mask], 98))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            fig.suptitle(f"FNO Autoencoder — test idx={idx}  (encode/decode, no diffusion)\n"
                        f"RMSE={rmse:.4f}  MagErr={mag_err:.4f}  AngErr={ang_err:.2f}°",
                        fontsize=11, fontweight="bold")
            im1 = plot_quiver(axes[0], true_np, land_mask, vmax_spd, "Ground Truth")
            plot_quiver(axes[1], pred_np, land_mask, vmax_spd, "Reconstruction")
            im2 = plot_error(axes[2], pred_np, true_np, land_mask, vmax_err,
                             f"|Error|  (RMSE={rmse:.4f})")
            fig.colorbar(im1, ax=axes[0:2], location="right", shrink=0.7, label="Speed", pad=0.01)
            fig.colorbar(im2, ax=axes[2], location="right", shrink=0.7, label="|error| speed", pad=0.01)

            out_path = os.path.join(args.out_dir, f"reconstruction_idx{idx:05d}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}\n")

    rmses = [r[1] for r in results]
    mags  = [r[2] for r in results]
    angs  = [r[3] for r in results]
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"FNO Autoencoder — plain encode/decode inference (no diffusion)\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"N samples  : {len(results)}\n\n")
        f.write(f"{'idx':>6} {'RMSE':>10} {'MagErr':>10} {'AngErr(deg)':>12}\n")
        f.write("-" * 42 + "\n")
        for idx, rmse, mag_err, ang_err in results:
            f.write(f"{idx:>6} {rmse:>10.5f} {mag_err:>10.5f} {ang_err:>12.3f}\n")
        f.write("-" * 42 + "\n")
        f.write(f"{'mean':>6} {np.mean(rmses):>10.5f} {np.mean(mags):>10.5f} {np.mean(angs):>12.3f}\n")

    print(f"Summary saved: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
