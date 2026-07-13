"""
batch_rmse_dps.py  –  Batch RMSE evaluation using DPS (ζ=0.04) on the baseline model.
Runs 20 test samples through DPS inference and writes results alongside the existing
repaint results for easy comparison.
"""
import argparse
import os
import pickle
import numpy as np
import torch

from diffusion import DDPM
from repaint_model import Repaint


# ── shared path generator (same as batch_rmse_repaint_r1.py) ─────────────────

def biased_walk_path(land_mask, n_steps=150, seed=None, straight_bias=0.75):
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape
    ocean_cells = list(zip(*np.where(~land_mask)))
    if not ocean_cells:
        raise ValueError("No ocean cells found in land_mask")
    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c = int(start[0]), int(start[1])
    path_mask = np.zeros((H, W), dtype=bool)
    path_mask[r, c] = True
    all_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cur_dir = all_dirs[rng.integers(4)]
    visit_count = np.zeros((H, W), dtype=np.float32)
    visit_count[r, c] = 1.0
    for _ in range(n_steps - 1):
        valid = [
            (dr, dc)
            for dr, dc in all_dirs
            if 0 <= r + dr < H and 0 <= c + dc < W and not land_mask[r + dr, c + dc]
        ]
        if not valid:
            break
        side = (1.0 - straight_bias) / 2.0
        weights = []
        for dr, dc in valid:
            dot = dr * cur_dir[0] + dc * cur_dir[1]
            w = straight_bias if dot == 1 else (side if dot == 0 else side * 0.05)
            nr, nc = r + dr, c + dc
            weights.append(w / (1.0 + visit_count[nr, nc]))
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        idx = rng.choice(len(valid), p=weights)
        dr, dc = valid[idx]
        r, c = r + dr, c + dc
        cur_dir = (dr, dc)
        visit_count[r, c] += 1.0
        path_mask[r, c] = True
    return path_mask


# ── DPS inference ─────────────────────────────────────────────────────────────

def dps_infer(model, diffusion, x0_known, path_mask, land_mask,
              device="cpu", step_size=0.04):
    """
    Diffusion Posterior Sampling (DPS) for inpainting.
    Chung et al. (2022), Algorithm 1.
    step_size = ζ  (gradient guidance scale, normalised by ||residual||)
    """
    H, W = x0_known.shape[1:]
    x0_known_t = x0_known.unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    T  = diffusion.T

    for t_int in reversed(range(T)):
        t_prev_int = max(t_int - 1, 0)

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        # (1) Predict noise and Tweedie x̂₀
        pred_noise = model(xt_in, t_vec)
        ab         = diffusion.alpha_bar[t_int]
        x0_hat     = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat     = x0_hat.clamp(-3.0, 3.0)

        # (2) Measurement residual at path cells only
        residual = known_t * (x0_hat - x0_known_t)
        norm_sq  = (residual ** 2).sum()

        # (3) Gradient of ||residual||^2 w.r.t. x_t
        grad = torch.autograd.grad(norm_sq, xt_in)[0]

        # (4) Standard DDPM reverse step + DPS correction
        with torch.no_grad():
            xt_next  = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)
            norm     = norm_sq.sqrt().item() + 1e-8
            xt_next  = xt_next - (step_size / norm) * grad.detach()
            xt_next  = xt_next * ocean_t

        xt = xt_next

    return xt.squeeze(0).cpu().numpy()


# ── helpers ───────────────────────────────────────────────────────────────────

def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask]) ** 2)))


def load_split(data, split_name):
    SPLIT_IDX = {"train": 0, "val": 1, "test": 2}
    key = split_name if (isinstance(data, dict) and split_name in data) else SPLIT_IDX[split_name]
    arr    = np.asarray(data[key], dtype=np.float32)      # (H, W, 2, N)
    fields = np.transpose(arr, (3, 2, 0, 1)).astype(np.float32)
    land   = np.isnan(arr[:, :, 0, 0])
    return np.nan_to_num(fields, nan=0.0), land


# ── main evaluation ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",    default="/root/ocean_ddpm/data.pickle")
    p.add_argument("--ckpt",      default="/root/autoencoder_train/checkpoints_linear/best_model_linear.pt",
                   help="Checkpoint to evaluate (any Repaint/DDPM model)")
    p.add_argument("--base_ckpt", default=None, help="Alias for --ckpt (backwards compat)")
    p.add_argument("--label",     default=None,
                   help="Label for output files (default: auto from ckpt filename)")
    p.add_argument("--out_dir",   default="/root/autoencoder_train/inference_results")
    p.add_argument("--n_samples", type=int,   default=20)
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--path_steps",   type=int, default=150)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--zeta",         type=float, default=0.04,
                   help="DPS gradient step size (ζ)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve ckpt path (--base_ckpt takes priority for backwards compat)
    ckpt_path = args.base_ckpt if args.base_ckpt else args.ckpt
    label     = args.label or os.path.splitext(os.path.basename(ckpt_path))[0]
    label_safe = label.replace(" ", "_")

    print(f"Device  : {device}")
    print(f"ζ (zeta): {args.zeta}")
    print(f"Ckpt    : {ckpt_path}")
    print(f"Label   : {label_safe}")

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    train_fields, train_land = load_split(data, "train")
    test_fields,  test_land  = load_split(data, "test")
    land_mask  = test_land
    ocean_mask = ~land_mask

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    T         = ckpt_args.get("T", 1000)
    schedule  = ckpt.get("schedule", "linear")
    noise_std = ckpt.get("noise_std") or float(train_fields[:, :, ocean_mask].std())
    print(f"T={T}  schedule={schedule}  noise_std={noise_std:.5f}")

    model = Repaint(in_ch=2, base_ch=ckpt_args.get("base_ch", 64),
                    time_dim=ckpt_args.get("time_dim", 256)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = DDPM(T=T, beta_schedule=schedule, device=device, noise_std=noise_std)
    method_name = f"dps_z{str(args.zeta).replace('.','')[:5]}_{label_safe[:20]}"

    n_samples = min(args.n_samples, test_fields.shape[0] - args.sample_start)
    idxs = list(range(args.sample_start, args.sample_start + n_samples))

    rmses = []
    rows  = [f"sample_idx,{method_name}"]

    for c, idx in enumerate(idxs, start=1):
        true      = test_fields[idx]
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps,
                                     seed=args.seed + idx)
        x_obs          = true.copy()
        x_obs[:, ~path_mask] = 0.0

        pred = dps_infer(model, diffusion,
                         torch.from_numpy(x_obs),
                         path_mask, land_mask,
                         device=device, step_size=args.zeta)

        rmse = rmse_ocean(pred, true, ocean_mask)
        rmses.append(rmse)
        rows.append(f"{idx},{rmse:.8f}")

        if c % 5 == 0 or c == n_samples:
            print(f"  [{c}/{n_samples}]  sample {idx}  RMSE={rmse:.4f}")

    arr  = np.array(rmses)
    mean, std = float(arr.mean()), float(arr.std())
    mn,   mx  = float(arr.min()),  float(arr.max())

    summary = (
        f"DPS Evaluation (ζ={args.zeta})\n"
        f"n_samples={n_samples}  path_steps={args.path_steps}  seed={args.seed}\n"
        f"checkpoint : {ckpt_path}\n"
        f"label      : {label_safe}\n\n"
        f"{'Method':<40} {'Mean RMSE':>12} {'Std':>10} {'Min':>10} {'Max':>10}\n"
        f"{'-'*80}\n"
        f"{method_name:<40} {mean:12.6f} {std:10.6f} {mn:10.6f} {mx:10.6f}\n"
    )
    print("\n" + summary)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"dps_{label_safe}_z{str(args.zeta).replace('.','')[:5]}"
    with open(os.path.join(args.out_dir, f"{tag}_summary.txt"), "w") as f:
        f.write(summary)
    with open(os.path.join(args.out_dir, f"{tag}_per_sample.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Saved to {args.out_dir}/{tag}_summary.txt")


if __name__ == "__main__":
    main()
