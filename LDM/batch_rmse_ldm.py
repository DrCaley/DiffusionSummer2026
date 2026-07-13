"""
batch_rmse_ldm.py  —  Latent Diffusion Model inference with DPS conditioning.

Algorithm (Latent DPS through decoder):
  At each reverse step t:
  1. Latent DDPM forward:  z_t → eps_pred → z0_hat  (Tweedie)
  2. VAE decode:           z0_hat → x0_hat  (pixel space)
  3. Measurement residual: r = path_mask * (x0_hat - x0_obs)
  4. DPS correction:       grad = d||r||² / dz_t   (autograd through decoder)
  5. Reverse step:         z_{t-1} = DDPM_reverse(z_t) - (ζ/||r||) * grad

Final output: VAE.decode(z_0)

This is genuinely stochastic — starts from random z_T in the 4×24×12 latent space.
Observations are enforced through DPS guidance in pixel space.
"""
import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from diffusion import DDPM
from vae_model import OceanVAE
from latent_unet import LatentUNet


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


# ── latent DPS inference ───────────────────────────────────────────────────

def latent_dps_infer(vae, latent_model, diffusion,
                     x0_obs_np, path_mask, land_mask,
                     device, zeta=0.04):
    """
    Full latent DPS:
      - Reverse chain in latent space (4, 24, 12)
      - DPS gradient backpropagated through VAE decoder
      - Conditioning on sparse path-cell observations in pixel space
    """
    H, W = land_mask.shape
    x0_obs  = torch.from_numpy(x0_obs_np).unsqueeze(0).to(device)
    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    # Sample initial latent from prior N(0, noise_std²·I)
    C, Hl, Wl = vae.latent_shape
    zt = torch.randn(1, C, Hl, Wl, device=device) * diffusion.noise_std

    vae.eval(); latent_model.eval()

    for t_int in reversed(range(diffusion.T)):
        t_prev = max(t_int - 1, 0)

        zt_in = zt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        # (1) Latent DDPM: Tweedie estimate of z_0
        eps_pred = latent_model(zt_in, t_vec)
        ab       = diffusion.alpha_bar[t_int]
        z0_hat   = (zt_in - (1.0 - ab).sqrt() * eps_pred) / ab.sqrt()
        z0_hat   = z0_hat.clamp(-5.0, 5.0)

        # (2) Decode to pixel space
        x0_hat = vae.decode(z0_hat, orig_H=H, orig_W=W)   # (1, 2, H, W)
        x0_hat = x0_hat * ocean_t

        # (3) Measurement residual at path cells
        residual = known_t * (x0_hat - x0_obs)
        norm_sq  = (residual ** 2).sum()

        # (4) Gradient w.r.t. z_t through decoder
        grad = torch.autograd.grad(norm_sq, zt_in)[0]

        # (5) DDPM reverse step in latent space + DPS correction
        with torch.no_grad():
            zt_next  = diffusion.p_sample_step(latent_model, zt_in.detach(), t_int, t_prev)
            norm     = norm_sq.sqrt().item() + 1e-8
            zt_next  = zt_next - (zeta / norm) * grad.detach()

        zt = zt_next

    # Final decode
    with torch.no_grad():
        x0_final = vae.decode(zt, orig_H=H, orig_W=W) * ocean_t

    return x0_final.squeeze(0).cpu().numpy()


# ── helpers ─────────────────────────────────────────────────────────────────

def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask])**2)))


def load_split(data, split_name):
    IDX = {"train": 0, "val": 1, "test": 2}
    key = split_name if (isinstance(data, dict) and split_name in data) else IDX[split_name]
    arr = np.asarray(data[key], dtype=np.float32)
    return np.nan_to_num(np.transpose(arr, (3,2,0,1)).astype(np.float32)), np.isnan(arr[:,:,0,0])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--vae_ckpt",    default="/root/ldm/checkpoints_vae/best_vae.pt")
    p.add_argument("--ldm_ckpt",    default="/root/ldm/checkpoints_ldm/best_latent_ddpm.pt")
    p.add_argument("--out_dir",     default="/root/autoencoder_train/inference_results")
    p.add_argument("--n_samples",   type=int,   default=50)
    p.add_argument("--sample_start",type=int,   default=0)
    p.add_argument("--path_steps",  type=int,   default=150)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--zeta",        type=float, default=0.04)
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"ζ      : {args.zeta}")

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    _, train_land = load_split(data, "train")
    test_fields, land_mask = load_split(data, "test")
    ocean_mask = ~land_mask

    # ── Load VAE ──────────────────────────────────────────────────────────
    vae_ck   = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_args = vae_ck.get("args", {})
    vae = OceanVAE(c_lat=vae_args.get("c_lat", 4),
                   base_ch=vae_args.get("base_ch", 32)).to(device)
    vae.load_state_dict(vae_ck["model"]); vae.eval()
    print(f"VAE loaded. Latent shape: {vae.latent_shape}")

    # ── Load latent DDPM ──────────────────────────────────────────────────
    ldm_ck   = torch.load(args.ldm_ckpt, map_location=device, weights_only=False)
    ldm_args = ldm_ck.get("args", {})
    c_lat    = ldm_ck.get("c_lat", vae.c_lat)
    noise_std= ldm_ck.get("noise_std", 1.0)
    latent_m = LatentUNet(in_ch=c_lat,
                          base_ch=ldm_args.get("base_ch", 64),
                          time_dim=ldm_args.get("time_dim", 256)).to(device)
    latent_m.load_state_dict(ldm_ck["model"]); latent_m.eval()
    diffusion = DDPM(T=ldm_args.get("T", 1000),
                     beta_schedule=ldm_args.get("schedule", "linear"),
                     device=device, noise_std=noise_std)
    print(f"Latent DDPM loaded. T={diffusion.T}, noise_std={noise_std:.5f}")

    n_samples = min(args.n_samples, test_fields.shape[0] - args.sample_start)
    idxs = list(range(args.sample_start, args.sample_start + n_samples))

    rmses = []
    rows  = ["sample_idx,ldm_dps"]

    for c, idx in enumerate(idxs, start=1):
        true      = test_fields[idx]
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps,
                                     seed=args.seed + idx)
        x_obs = true.copy(); x_obs[:, ~path_mask] = 0.0

        pred = latent_dps_infer(vae, latent_m, diffusion,
                                 x_obs, path_mask, land_mask,
                                 device=device, zeta=args.zeta)

        rmse = rmse_ocean(pred, true, ocean_mask)
        rmses.append(rmse); rows.append(f"{idx},{rmse:.8f}")

        if c % 5 == 0 or c == n_samples:
            print(f"  [{c}/{n_samples}]  sample {idx}  RMSE={rmse:.4f}")

    arr  = np.array(rmses)
    mean, std = arr.mean(), arr.std()
    mn, mx    = arr.min(), arr.max()

    summary = (
        f"Latent DPS Evaluation\n"
        f"n_samples={n_samples}  path_steps={args.path_steps}  seed={args.seed}\n"
        f"ζ={args.zeta}  T={diffusion.T}\n\n"
        f"{'Method':<20} {'Mean RMSE':>12} {'Std':>10} {'Min':>10} {'Max':>10}\n"
        f"{'-'*66}\n"
        f"{'ldm_dps':<20} {mean:12.6f} {std:10.6f} {mn:10.6f} {mx:10.6f}\n"
    )
    print("\n" + summary)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ldm_dps_summary.txt"), "w") as f:
        f.write(summary)
    with open(os.path.join(args.out_dir, "ldm_dps_per_sample.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Saved to {args.out_dir}/ldm_dps_summary.txt")


if __name__ == "__main__":
    main()
