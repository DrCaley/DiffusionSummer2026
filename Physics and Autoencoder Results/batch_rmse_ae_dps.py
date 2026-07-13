"""
batch_rmse_ae_dps.py
AE-guided DPS: uses AE prediction as a soft prior inside the DPS gradient,
alongside the sparse path observation constraint.

At each reverse step t:
  1. Run model to get x0_hat = Tweedie(x_t)
  2. Compute combined loss:
       L = ||path_mask * (x0_hat - x0_obs)||²  +  λ * ||(x0_hat - x0_ae)||²
  3. grad = ∂L/∂x_t  (autograd)
  4. x_{t-1} = DDPM_reverse(x_t) - (ζ / ||path_residual||) * grad

x0_ae is computed once (AE forward pass) before the reverse chain begins.
No training required — uses existing AE and diffusion checkpoints.

Sweeps λ over [0.1, 0.5, 1.0] on both base and physics models.
"""
import argparse
import os
import pickle
import numpy as np
import torch

from diffusion import DDPM
from repaint_model import Repaint
from ae_model import RepaintAutoencoder


# ── path generator ─────────────────────────────────────────────────────────

def biased_walk_path(land_mask, n_steps=150, seed=None, straight_bias=0.75):
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape
    ocean_cells = list(zip(*np.where(~land_mask)))
    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c  = int(start[0]), int(start[1])
    path_mask = np.zeros((H, W), dtype=bool); path_mask[r, c] = True
    all_dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    cur_dir  = all_dirs[rng.integers(4)]
    vc = np.zeros((H, W), dtype=np.float32); vc[r, c] = 1.0
    for _ in range(n_steps - 1):
        valid = [(dr,dc) for dr,dc in all_dirs
                 if 0<=r+dr<H and 0<=c+dc<W and not land_mask[r+dr,c+dc]]
        if not valid: break
        side = (1.0 - straight_bias) / 2.0
        weights = []
        for dr, dc in valid:
            dot = dr*cur_dir[0] + dc*cur_dir[1]
            w = straight_bias if dot==1 else (side if dot==0 else side*0.05)
            weights.append(w / (1.0 + vc[r+dr, c+dc]))
        weights = np.array(weights); weights /= weights.sum()
        idx = rng.choice(len(valid), p=weights)
        dr, dc = valid[idx]; r, c = r+dr, c+dc
        cur_dir = (dr, dc); vc[r, c] += 1.0; path_mask[r, c] = True
    return path_mask


# ── AE-guided DPS ─────────────────────────────────────────────────────────

def ae_guided_dps(ae_model, diff_model, diffusion,
                  x0_known_np, path_mask, land_mask,
                  device, zeta=0.04, lambda_ae=0.5):
    """
    Full-chain DPS (T=1000 steps) guided by both sparse observations
    and the AE reconstruction as a soft prior.
    """
    H, W = land_mask.shape
    x0_known_t = torch.from_numpy(x0_known_np).unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    # ── Step 1: Compute AE prior (once, before reverse chain) ────────────
    x_obs   = x0_known_np.copy(); x_obs[:, ~path_mask] = 0.0
    mask_ch = path_mask.astype(np.float32)[None]
    ae_inp  = torch.from_numpy(np.concatenate([x_obs, mask_ch], axis=0)).unsqueeze(0).to(device)
    ae_model.eval()
    with torch.no_grad():
        x0_ae = ae_model(ae_inp) * ocean_t   # (1, 2, H, W)

    # ── Step 2: Reverse chain with dual-measurement DPS ──────────────────
    diff_model.eval()
    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t

    for t_int in reversed(range(diffusion.T)):
        t_prev = max(t_int - 1, 0)

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        # Predict noise → Tweedie x̂₀
        pred_noise = diff_model(xt_in, t_vec)
        ab         = diffusion.alpha_bar[t_int]
        x0_hat     = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat     = x0_hat.clamp(-3.0, 3.0)

        # Dual-measurement loss
        res_obs = known_t * (x0_hat - x0_known_t)        # path observation
        res_ae  = x0_hat - x0_ae                          # AE prior
        loss    = (res_obs**2).sum() + lambda_ae * (res_ae**2).sum()
        norm    = (res_obs**2).sum().sqrt().item() + 1e-8

        grad = torch.autograd.grad(loss, xt_in)[0]

        with torch.no_grad():
            xt_next  = diffusion.p_sample_step(diff_model, xt_in.detach(), t_int, t_prev)
            xt_next  = xt_next - (zeta / norm) * grad.detach()
            xt_next  = xt_next * ocean_t

        xt = xt_next

    return xt.squeeze(0).cpu().numpy()


# ── helpers ─────────────────────────────────────────────────────────────────

def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask])**2)))


def magnitude_rmse_ocean(pred, true, ocean_mask):
    """RMSE of speed (scalar magnitude) over ocean cells."""
    spd_pred = np.sqrt(pred[0]**2 + pred[1]**2)
    spd_true = np.sqrt(true[0]**2 + true[1]**2)
    return float(np.sqrt(np.mean((spd_pred[ocean_mask] - spd_true[ocean_mask])**2)))


def angle_error_ocean(pred, true, ocean_mask, min_speed=0.01):
    """
    Mean angle error (degrees) over ocean cells where |true| >= min_speed.
    Uses arccos(dot / (|pred||true|)), clipped to [-1, 1] for numerical safety.
    """
    u_p, v_p = pred[0][ocean_mask], pred[1][ocean_mask]
    u_t, v_t = true[0][ocean_mask], true[1][ocean_mask]
    spd_t = np.sqrt(u_t**2 + v_t**2)
    spd_p = np.sqrt(u_p**2 + v_p**2)
    mask  = spd_t >= min_speed
    if not mask.any():
        return float("nan")
    dot   = (u_p[mask]*u_t[mask] + v_p[mask]*v_t[mask])
    denom = (spd_p[mask] * spd_t[mask]).clip(min=1e-8)
    cos_a = np.clip(dot / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a).mean()))


def load_split(data, split_name):
    IDX = {"train": 0, "val": 1, "test": 2}
    key = split_name if (isinstance(data, dict) and split_name in data) else IDX[split_name]
    arr = np.asarray(data[key], dtype=np.float32)
    return np.nan_to_num(np.transpose(arr, (3,2,0,1)).astype(np.float32)), np.isnan(arr[:,:,0,0])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",       default="/root/ocean_ddpm/data.pickle")
    p.add_argument("--ae_ckpt",      default="/root/autoencoder_train/checkpoints/best_model_autoencoder.pt")
    p.add_argument("--base_ckpt",    default="/root/autoencoder_train/checkpoints_linear/best_model_linear.pt")
    p.add_argument("--physics_ckpt", default="/root/autoencoder_train/checkpoints_physics/best_model_physics.pt")
    p.add_argument("--out_dir",      default="/root/autoencoder_train/inference_results")
    p.add_argument("--n_samples",    type=int,   default=50)
    p.add_argument("--sample_start", type=int,   default=0)
    p.add_argument("--path_steps",   type=int,   default=150)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--zeta",         type=float, default=0.04)
    p.add_argument("--lambdas",      default="0.1,0.5,1.0",
                   help="Comma-separated λ_ae values to sweep")
    p.add_argument("--models",       default="base,physics",
                   help="Comma-separated models to evaluate: base, physics")
    p.add_argument("--device",       default=None)
    args = p.parse_args()

    device   = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lambdas  = [float(x) for x in args.lambdas.split(",")]
    models   = [x.strip() for x in args.models.split(",")]
    print(f"Device  : {device}")
    print(f"ζ       : {args.zeta}")
    print(f"λ sweep : {lambdas}")
    print(f"models  : {models}")

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    train_fields, _         = load_split(data, "train")
    test_fields,  land_mask = load_split(data, "test")
    ocean_mask = ~land_mask

    # ── Load AE ──────────────────────────────────────────────────────────
    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae_m  = RepaintAutoencoder(in_ch=3, out_ch=2,
                                base_ch=ae_ck.get("args", {}).get("base_ch", 64)).to(device)
    ae_m.load_state_dict(ae_ck["model"]); ae_m.eval()
    print(f"AE loaded")

    # ── Load diffusion models ─────────────────────────────────────────────
    def load_ddpm(ckpt_path, name):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        ca = ck.get("args", {})
        ns = ck.get("noise_std") or float(train_fields[:, :, ocean_mask].std())
        m  = Repaint(in_ch=2, base_ch=ca.get("base_ch", 64),
                     time_dim=ca.get("time_dim", 256)).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        d  = DDPM(T=ca.get("T", 1000), beta_schedule=ck.get("schedule", "linear"),
                  device=device, noise_std=ns)
        print(f"{name}: T={d.T} noise_std={ns:.5f}")
        return m, d

    diff_models = {}
    if "base" in models:
        diff_models["base"] = load_ddpm(args.base_ckpt, "Base")
    if "physics" in models:
        diff_models["physics"] = load_ddpm(args.physics_ckpt, "Physics")

    # Build variant keys: ae_dps_base_l0.5, ae_dps_physics_l0.5, etc.
    variants = []
    for mname, (dm, diff) in diff_models.items():
        for lam in lambdas:
            key = f"ae_dps_{mname}_l{lam}"
            variants.append((key, dm, diff, lam))

    n_samples = min(args.n_samples, test_fields.shape[0] - args.sample_start)
    idxs = list(range(args.sample_start, args.sample_start + n_samples))

    results  = {k: {"rmse": [], "mag_rmse": [], "angle_err": []} for k, *_ in variants}
    csv_rows = {k: [f"sample_idx,rmse,mag_rmse,angle_err_deg"] for k, *_ in variants}

    for c, idx in enumerate(idxs, start=1):
        true      = test_fields[idx]
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps,
                                     seed=args.seed + idx)
        x_obs = true.copy(); x_obs[:, ~path_mask] = 0.0

        for key, dm, diff, lam in variants:
            pred = ae_guided_dps(ae_m, dm, diff, x_obs, path_mask, land_mask,
                                 device=device, zeta=args.zeta, lambda_ae=lam)
            rmse   = rmse_ocean(pred, true, ocean_mask)
            mag_e  = magnitude_rmse_ocean(pred, true, ocean_mask)
            ang_e  = angle_error_ocean(pred, true, ocean_mask)
            results[key]["rmse"].append(rmse)
            results[key]["mag_rmse"].append(mag_e)
            results[key]["angle_err"].append(ang_e)
            csv_rows[key].append(f"{idx},{rmse:.8f},{mag_e:.8f},{ang_e:.4f}")

        if c % 5 == 0 or c == n_samples:
            print(f"  [{c}/{n_samples}]", end="")
            for key, *_ in variants:
                print(f"  {key}=rmse:{results[key]['rmse'][-1]:.4f} ang:{results[key]['angle_err'][-1]:.1f}°", end="")
            print()

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    lines = [
        "AE-Guided DPS Evaluation",
        f"n_samples={n_samples}  path_steps={args.path_steps}  seed={args.seed}",
        f"ζ={args.zeta}  λ sweep={lambdas}  models={models}",
        "",
        f"{'Method':<36} {'Mean RMSE':>10} {'Std':>8} {'Mag RMSE':>10} {'Angle Err\u00b0':>11} {'Min RMSE':>10} {'Max RMSE':>10}",
        "-" * 100,
    ]
    csv_summary = ["model,mean_rmse,std_rmse,mag_rmse,angle_err_deg,min_rmse,max_rmse,n_samples"]

    for key, *_ in variants:
        rmse_arr = np.array(results[key]["rmse"])
        mag_arr  = np.array(results[key]["mag_rmse"])
        ang_arr  = np.array([x for x in results[key]["angle_err"] if not np.isnan(x)])
        m, s, mn, mx = rmse_arr.mean(), rmse_arr.std(), rmse_arr.min(), rmse_arr.max()
        mag_m = mag_arr.mean()
        ang_m = ang_arr.mean() if len(ang_arr) else float("nan")
        lines.append(f"{key:<36} {m:10.6f} {s:8.6f} {mag_m:10.6f} {ang_m:11.2f} {mn:10.6f} {mx:10.6f}")
        csv_summary.append(f"{key},{m:.8f},{s:.8f},{mag_m:.8f},{ang_m:.4f},{mn:.8f},{mx:.8f},{n_samples}")
        with open(os.path.join(args.out_dir, f"{key}_per_sample.csv"), "w") as f:
            f.write("\n".join(csv_rows[key]) + "\n")

    summary_str = "\n".join(lines)
    print("\n" + summary_str)
    with open(os.path.join(args.out_dir, "ae_dps_summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    with open(os.path.join(args.out_dir, "ae_dps_summary.csv"), "w") as f:
        f.write("\n".join(csv_summary) + "\n")
    print(f"\nSaved to {args.out_dir}/ae_dps_summary.txt")


if __name__ == "__main__":
    main()
