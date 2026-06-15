"""
batch_infer.py – Batch RePaint/DPS inference over N validation samples.

Reports mean RMSE ± std and mean MAE ± std over all evaluated samples.
Results images are saved to testing/results/.

Usage
-----
cd DDPM
python batch_infer.py \\
    --checkpoint checkpoints/model_DDPM_MSE_coloredGaussian_cosine.pt \\
    --pickle ../data.pickle \\
    --n 10 \\
    --method repaint \\
    --path_steps 150 \\
    --resample 10
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_data
from model.unet import UNet
from testing.repaint.repaint_infer import repaint_sample
from testing.DPS.dps_infer import dps_sample
from visualize_infer import simulate_robot_path, plot_field, plot_error


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits, land_mask = load_data(args.pickle)
    land_np  = land_mask.numpy()
    ocean_np = ~land_np

    val_data = splits[1]
    n        = min(args.n, len(val_data))

    # ---- Load model ----
    ckpt       = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = UNet(
        in_channels=2,
        base_ch=saved_args.get("base_ch", 128),
        ch_mults=tuple(saved_args.get("ch_mults", [1, 2, 2])),
        time_dim=saved_args.get("time_dim", 512),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    noise_alpha = saved_args.get("noise_alpha", 2.0)
    print(f"Loaded epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.5f}")

    os.makedirs(args.out_dir, exist_ok=True)

    rmse_list = []
    mae_list  = []

    for i in range(n):
        rng     = np.random.default_rng(args.seed + i)
        x0_np   = val_data[i]                         # (2, H, W)
        path    = simulate_robot_path(ocean_np, args.path_steps, rng=rng)

        known_mask_np = np.zeros(x0_np.shape[1:], dtype=bool)
        for r, c in path:
            known_mask_np[r, c] = True

        x0_t = torch.from_numpy(x0_np).unsqueeze(0).to(device)
        km_t = torch.from_numpy(known_mask_np)

        method = args.method.lower()
        if method == "repaint":
            x_pred_t = repaint_sample(
                model, x0_t, km_t, T=args.T, device=device,
                resample=args.resample, noise_alpha=noise_alpha,
            )
        else:
            x_pred_t = dps_sample(
                model, x0_t, km_t, T=args.T, device=device,
                step_size=args.dps_step, noise_alpha=noise_alpha,
            )

        x0_pred_np = x_pred_t[0].cpu().numpy()
        diff       = x0_pred_np - x0_np
        diff[:, land_np] = 0.0
        ocean_diff = diff[:, ocean_np]
        rmse = float(np.sqrt((ocean_diff ** 2).mean()))
        mae  = float(np.abs(ocean_diff).mean())
        rmse_list.append(rmse)
        mae_list.append(mae)

        print(f"  Sample {i:3d}  RMSE={rmse:.4f}  MAE={mae:.4f}")

        # Save individual figure
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        pct = 100 * known_mask_np.sum() / ocean_np.sum()
        fig.suptitle(
            f"Colored Gaussian Noise DDPM ({method.upper()})  |  Val sample {i}\n"
            f"RMSE = {rmse:.4f}   MAE = {mae:.4f}   Path = {pct:.1f}%",
            fontsize=11,
        )
        u_gt, v_gt  = x0_np[0],      x0_np[1]
        u_pr, v_pr  = x0_pred_np[0], x0_pred_np[1]
        vmax_gt     = float(np.nanpercentile(np.hypot(u_gt[ocean_np], v_gt[ocean_np]), 98))
        plot_field(axes[0], u_gt, v_gt, land_np, "Ground Truth", vmax=vmax_gt)
        plot_field(axes[1], u_pr, v_pr, land_np,
                   f"Prediction + robot path", vmax=vmax_gt, path=path)
        plot_error(axes[2], diff[0], diff[1], land_np,
                   f"|Error|  RMSE={rmse:.4f}")
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"sample_{i:03d}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ---- Summary ----
    rmse_arr = np.array(rmse_list)
    mae_arr  = np.array(mae_list)
    print(f"\n{'='*50}")
    print(f"Batch evaluation  ({method.upper()}, n={n})")
    print(f"  Mean RMSE : {rmse_arr.mean():.4f} ± {rmse_arr.std():.4f}")
    print(f"  Mean MAE  : {mae_arr.mean():.4f}  ± {mae_arr.std():.4f}")
    print(f"  Results saved to: {args.out_dir}")

    # Save summary CSV
    csv_path = os.path.join(args.out_dir, "batch_results.csv")
    with open(csv_path, "w") as fh:
        fh.write("sample,rmse,mae\n")
        for i, (r, m) in enumerate(zip(rmse_list, mae_list)):
            fh.write(f"{i},{r:.6f},{m:.6f}\n")
        fh.write(f"mean,{rmse_arr.mean():.6f},{mae_arr.mean():.6f}\n")
        fh.write(f"std,{rmse_arr.std():.6f},{mae_arr.std():.6f}\n")
    print(f"  CSV → {csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pickle",     default="../data.pickle")
    p.add_argument("--n",          type=int,   default=10)
    p.add_argument("--method",     default="repaint", choices=["repaint", "dps"])
    p.add_argument("--path_steps", type=int,   default=150)
    p.add_argument("--resample",   type=int,   default=10)
    p.add_argument("--dps_step",   type=float, default=1.0)
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out_dir",    default="testing/results")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
