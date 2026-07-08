"""
Compute Schedule Deviation diagnostic for a RePaint-style conditional model.

This script loads a trained checkpoint (same format used by
`batch_stride_infer.py`), runs RePaint inference while recording the
per-timestep denoised `x0_pred` estimates output by the model, and
computes a simple Schedule Deviation metric:

  SD = mean_t ||x0_pred(t) - x0_pred(t_prev)||_2 / (||x0_pred(t_prev)||_2 + eps)

which measures how rapidly the model's denoised prediction changes
between successive reverse steps. The script writes a per-sample
text report and optional numpy arrays for deeper analysis / plotting.

Usage (from workspace root):
    python "Repaint vs DPS/schedule_deviation.py" --checkpoint <ckpt> --n_samples 5

"""

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import glob

# If there are local copies of diffusion.py / repaint_model.py in subfolders
# (e.g., packaged model directory), add their parent folder to sys.path so
# the script can import them.
local_diff = glob.glob(os.path.join(os.path.dirname(__file__), "**", "diffusion.py"), recursive=True)
if local_diff:
    local_dir = os.path.dirname(local_diff[0])
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)

local_repaint = glob.glob(os.path.join(os.path.dirname(__file__), "**", "repaint_model.py"), recursive=True)
if local_repaint:
    local_dir2 = os.path.dirname(local_repaint[0])
    if local_dir2 not in sys.path:
        sys.path.insert(0, local_dir2)

# Find a repo-level loss_functions.py and add its directory to sys.path if found.
loss_paths = glob.glob(os.path.join(os.path.dirname(__file__), "..", "**", "loss_functions.py"), recursive=True)
if loss_paths:
    loss_dir = os.path.dirname(loss_paths[0])
    if loss_dir not in sys.path:
        sys.path.insert(0, loss_dir)

from utils.dataset import OceanCurrentDataset
from diffusion import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path


def compute_x0_pred_from_noise(xt: torch.Tensor, pred_noise: torch.Tensor, t_int: int, diffusion: DDPM):
    """Reproduce x0_pred calculation from diffusion.p_sample_step."""
    device = xt.device
    ab = diffusion.alpha_bar[t_int].to(device)
    sqrt_mab = (1.0 - ab).sqrt()
    sqrt_ab = ab.sqrt()
    x0_pred = (xt - sqrt_mab * pred_noise) / sqrt_ab
    return x0_pred


@torch.no_grad()
def schedule_deviation_for_sample(
    model: torch.nn.Module,
    diffusion: DDPM,
    x0_known: torch.Tensor,
    path_mask: np.ndarray,
    land_mask: np.ndarray,
    r: int = 10,
    device: str = "cpu",
    stride: int = 1,
):
    model.eval()
    H, W = x0_known.shape[1:]
    x0_known = x0_known.unsqueeze(0).to(device)

    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t  = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std
    xt = xt * ocean_t
    T = diffusion.T

    timesteps = list(range(0, T, stride))

    x0_preds: List[torch.Tensor] = []

    for i in reversed(range(len(timesteps))):
        t_int = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        for j in range(r):
            B = xt.shape[0]
            t = torch.full((B,), t_int, device=device, dtype=torch.long)

            pred_noise = model(xt, t)
            x0_pred = compute_x0_pred_from_noise(xt, pred_noise, t_int, diffusion)
            # clamp as in training code to keep values bounded
            x0_pred = x0_pred.clamp(-1.5, 1.5)

            # Store a CPU copy for analysis
            x0_preds.append(x0_pred.squeeze(0).cpu().numpy())

            # Continue with RePaint merge/resample logic so xt evolves normally
            xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)

            t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_known_t, _ = diffusion.q_sample(x0_known, t_prev_t)

            xt_merged = known_t * xt_known_t + (1.0 - known_t) * xt_unknown
            xt_merged = xt_merged * ocean_t

            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int) * ocean_t
            else:
                xt = xt_merged

    # Compute deviation metric across successive stored x0_preds
    eps = 1e-8
    deviations = []
    for k in range(1, len(x0_preds)):
        a = x0_preds[k - 1]
        b = x0_preds[k]
        # mask ocean pixels
        ocean = (~land_mask).astype(float)
        diff = ((b - a) ** 2 * ocean[None, ...]).sum() ** 0.5
        denom = (a ** 2 * ocean[None, ...]).sum() ** 0.5 + eps
        deviations.append(float(diff / denom))

    mean_sd = float(np.mean(deviations)) if deviations else 0.0

    return {
        "mean_schedule_deviation": mean_sd,
        "per_step_deviation": deviations,
        "num_recorded": len(x0_preds),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Schedule Deviation diagnostic for RePaint models")
    p.add_argument("--pickle", default="data.pickle")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--resample", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--out", default=None, help="Output txt path")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.checkpoint is None:
        print("Error: --checkpoint required (point to training checkpoint .pt)")
        return

    # Data
    test_ds = OceanCurrentDataset(args.pickle, split=2)
    land_mask_np = test_ds.land_mask.numpy()

    # Model + checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch = ckpt_args.get("base_ch", 64)
    time_dim = ckpt_args.get("time_dim", 256)

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    noise_std = ckpt.get("noise_std", None)
    if noise_std is None:
        train_ds = OceanCurrentDataset(args.pickle, split=0)
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())

    beta_schedule = ckpt.get("schedule", "geometric")
    diffusion = DDPM(T=args.T, beta_schedule=beta_schedule, device=device, noise_std=noise_std)

    out_lines = []
    out_lines.append(f"Schedule Deviation Report for checkpoint: {args.checkpoint}")
    out_lines.append(f"Device: {device}")
    out_lines.append("")

    for idx in range(args.sample_start, args.sample_start + args.n_samples):
        x0_true = test_ds[idx]
        path_mask = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=42 + idx)

        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"Processing sample {idx}...")
        res = schedule_deviation_for_sample(
            model, diffusion, x0_obs, path_mask, land_mask_np,
            r=args.resample, device=device, stride=args.stride,
        )

        out_lines.append(f"sample {idx}: mean_SD={res['mean_schedule_deviation']:.6f}, records={res['num_recorded']}")
        # Optionally save per-step deviations for later plotting
        np.save(os.path.join(script_dir, f"sd_sample_{idx}_perstep.npy"), np.array(res["per_step_deviation"]))
        print(f"  mean_SD={res['mean_schedule_deviation']:.6f}  (saved per-step numpy)")

    if args.out is None:
        args.out = os.path.join(script_dir, "schedule_deviation_report.txt")

    with open(args.out, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()
