"""
batch_compare_stride10.py  —  Comprehensive comparison: stride=10, 4 methods
  1. Repaint r=1  (pixel-space, base DDPM)
  2. Repaint r=10 (pixel-space, base DDPM)
  3. Standard DPS ζ=0.04 (pixel-space, base DDPM)
  4. Latent DPS ζ=0.04 (VAE + latent DDPM)

Metrics per sample: RMSE, magnitude RMSE, angle error (°), time (s)
Also saves comparison images (6-panel per sample).
"""
import argparse
import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion import DDPM
from repaint_model import Repaint
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


# ── inference methods ──────────────────────────────────────────────────────

@torch.no_grad()
def repaint_infer(model, diffusion, x0_known, path_mask, land_mask,
                  device, stride=10, resample=1):
    """RePaint with configurable stride and resample count."""
    H, W       = x0_known.shape[1:]
    x0_known_t = x0_known.unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        for r_idx in range(resample):
            xt_u = diffusion.p_sample_step(model, xt, t_int, t_prev_int)
            tp   = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_k, _ = diffusion.q_sample(x0_known_t, tp)
            xt_merged = known_t * xt_k + (1.0 - known_t) * xt_u
            xt_merged = xt_merged * ocean_t

            if r_idx < resample - 1:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int) * ocean_t
            else:
                xt = xt_merged

    return xt.squeeze(0).cpu().numpy()


def dps_infer(model, diffusion, x0_known_np, path_mask, land_mask,
              device, stride=10, zeta=0.04):
    """Pixel-space DPS with configurable stride."""
    H, W       = x0_known_np.shape[1:]
    x0_known_t = torch.from_numpy(x0_known_np).unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        pred_noise = model(xt_in, t_vec)
        ab         = diffusion.alpha_bar[t_int]
        x0_hat     = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat     = x0_hat.clamp(-3.0, 3.0)

        residual = known_t * (x0_hat - x0_known_t)
        norm_sq  = (residual ** 2).sum()
        grad     = torch.autograd.grad(norm_sq, xt_in)[0]

        with torch.no_grad():
            xt_next  = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)
            norm     = norm_sq.sqrt().item() + 1e-8
            xt_next  = xt_next - (zeta / norm) * grad.detach()
            xt_next  = xt_next * ocean_t

        xt = xt_next

    return xt.squeeze(0).cpu().numpy()


def latent_dps_infer(vae, latent_model, diffusion,
                     x0_obs_np, path_mask, land_mask,
                     device, stride=10, zeta=0.04):
    """Latent DPS with configurable stride."""
    H, W    = land_mask.shape
    x0_obs  = torch.from_numpy(x0_obs_np).unsqueeze(0).to(device)
    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    C, Hl, Wl = vae.latent_shape
    zt = torch.randn(1, C, Hl, Wl, device=device) * diffusion.noise_std
    timesteps = list(range(0, diffusion.T, stride))

    vae.eval(); latent_model.eval()

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        zt_in = zt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        eps_pred = latent_model(zt_in, t_vec)
        ab       = diffusion.alpha_bar[t_int]
        z0_hat   = (zt_in - (1.0 - ab).sqrt() * eps_pred) / ab.sqrt()
        z0_hat   = z0_hat.clamp(-5.0, 5.0)

        x0_hat  = vae.decode(z0_hat, orig_H=H, orig_W=W) * ocean_t
        residual = known_t * (x0_hat - x0_obs)
        norm_sq  = (residual ** 2).sum()
        grad     = torch.autograd.grad(norm_sq, zt_in)[0]

        with torch.no_grad():
            zt_next  = diffusion.p_sample_step(latent_model, zt_in.detach(), t_int, t_prev_int)
            norm     = norm_sq.sqrt().item() + 1e-8
            zt_next  = zt_next - (zeta / norm) * grad.detach()

        zt = zt_next

    with torch.no_grad():
        x0_final = vae.decode(zt, orig_H=H, orig_W=W) * ocean_t
    return x0_final.squeeze(0).cpu().numpy()


# ── metrics ─────────────────────────────────────────────────────────────────

def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask])**2)))

def mag_rmse_ocean(pred, true, ocean_mask):
    sp = np.sqrt(pred[0]**2 + pred[1]**2)
    st = np.sqrt(true[0]**2 + true[1]**2)
    return float(np.sqrt(np.mean((sp[ocean_mask] - st[ocean_mask])**2)))

def angle_error_ocean(pred, true, ocean_mask, min_speed=0.01):
    up, vp = pred[0][ocean_mask], pred[1][ocean_mask]
    ut, vt = true[0][ocean_mask], true[1][ocean_mask]
    st = np.sqrt(ut**2 + vt**2)
    sp = np.sqrt(up**2 + vp**2)
    mask = st >= min_speed
    if not mask.any(): return float("nan")
    dot  = up[mask]*ut[mask] + vp[mask]*vt[mask]
    denom = (sp[mask]*st[mask]).clip(min=1e-8)
    return float(np.degrees(np.arccos(np.clip(dot/denom, -1, 1)).mean()))


# ── visualisation helpers ────────────────────────────────────────────────────

def plot_field(ax, field, land_mask, title, vmax=None, step=2, add_cbar=True):
    u, v = field[0].T, field[1].T
    lm   = land_mask.T
    H, W = u.shape
    speed = np.ma.masked_where(lm, np.sqrt(u**2 + v**2))
    if vmax is None:
        vmax = float(np.nanpercentile(speed.compressed(), 98)) if speed.count() else 1.0
    vmax = max(vmax, 1e-6)
    ext = [-0.5, W-0.5, -0.5, H-0.5]
    im = ax.imshow(speed, origin="lower", cmap="cool", vmin=0, vmax=vmax,
                   extent=ext, aspect="auto", zorder=0)
    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["none", "black"]),
              extent=ext, aspect="auto", zorder=1)
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq, vq = u[::step, ::step], v[::step, ::step]
    om     = ~lm[::step, ::step]
    ax.quiver(xq[om], yq[om], uq[om], vq[om],
              color="black", scale=12, width=0.003, zorder=2)
    if add_cbar:
        plt.colorbar(im, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=9); ax.set_xlabel("X"); ax.set_ylabel("Y")
    return vmax

def plot_path(ax, land_mask, path_mask, title):
    lm, pm = land_mask.T, path_mask.T
    H, W   = lm.shape
    ext    = [-0.5, W-0.5, -0.5, H-0.5]
    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["white", "black"]),
              extent=ext, aspect="auto", zorder=0)
    pd = np.zeros((H,W), dtype=float); pd[pm] = 1.0
    ax.imshow(pd, origin="lower", cmap="Reds", alpha=0.8,
              extent=ext, aspect="auto", zorder=1, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9); ax.set_xlabel("X"); ax.set_ylabel("Y")
    for patch, label in [("white","Ocean"),("#d62728","Path"),("black","Land")]:
        ax.legend(handles=[mpatches.Patch(facecolor=p, label=l)
                            for p, l in [("white","Ocean"),("#d62728","Path"),("black","Land")]],
                  loc="upper right", fontsize=7); break

def plot_error(ax, pred, true, land_mask, title, add_cbar=True):
    lm  = land_mask.T
    err = np.sqrt((pred[0]-true[0])**2 + (pred[1]-true[1])**2).T
    H, W = lm.shape
    ext  = [-0.5, W-0.5, -0.5, H-0.5]
    em   = np.ma.masked_where(lm, err)
    im   = ax.imshow(em, origin="lower", cmap="hot_r", aspect="auto", extent=ext, zorder=0)
    ax.imshow(lm, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["none","black"]),
              extent=ext, aspect="auto", zorder=1)
    if add_cbar:
        plt.colorbar(im, ax=ax, label="|error|", shrink=0.7)
    ax.set_title(title, fontsize=9); ax.set_xlabel("X"); ax.set_ylabel("Y")


def load_split(data, split_name):
    IDX = {"train": 0, "val": 1, "test": 2}
    key = split_name if (isinstance(data, dict) and split_name in data) else IDX[split_name]
    arr = np.asarray(data[key], dtype=np.float32)
    return np.nan_to_num(np.transpose(arr, (3,2,0,1)).astype(np.float32)), np.isnan(arr[:,:,0,0])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",      default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--base_ckpt",   default="/root/autoencoder_train/checkpoints_linear/best_model_linear.pt")
    p.add_argument("--vae_ckpt",    default="/root/ldm/checkpoints_vae/best_vae.pt")
    p.add_argument("--ldm_ckpt",    default="/root/ldm/checkpoints_ldm/best_latent_ddpm.pt")
    p.add_argument("--out_dir",     default="/root/ldm/stride10_comparison")
    p.add_argument("--n_samples",   type=int, default=50)
    p.add_argument("--sample_start",type=int, default=0)
    p.add_argument("--path_steps",  type=int, default=150)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--stride",      type=int, default=10)
    p.add_argument("--zeta",        type=float, default=0.04)
    p.add_argument("--img_idxs",    default="3,7,15,22,30",
                   help="Sample indices to save as images")
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    img_idxs = set(int(x) for x in args.img_idxs.split(","))
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    print(f"Device : {device}")
    print(f"Stride : {args.stride}  ({args.stride}x faster than full T=1000)")
    print(f"ζ      : {args.zeta}")

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    train_fields, train_land = load_split(data, "train")
    test_fields,  land_mask  = load_split(data, "test")
    ocean_mask = ~land_mask

    # ── Load base DDPM ────────────────────────────────────────────────────
    print("Loading base DDPM...")
    base_ck   = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    base_ca   = base_ck.get("args", {})
    noise_std = base_ck.get("noise_std") or float(train_fields[:, :, ocean_mask].std())
    base_m    = Repaint(in_ch=2, base_ch=base_ca.get("base_ch", 64),
                        time_dim=base_ca.get("time_dim", 256)).to(device)
    base_m.load_state_dict(base_ck["model"]); base_m.eval()
    base_diff = DDPM(T=base_ca.get("T", 1000), beta_schedule=base_ck.get("schedule", "linear"),
                     device=device, noise_std=noise_std)
    print(f"  noise_std={noise_std:.5f}")

    # ── Load VAE + Latent DDPM ────────────────────────────────────────────
    print("Loading VAE + Latent DDPM...")
    vae_ck  = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_a   = vae_ck.get("args", {})
    vae     = OceanVAE(c_lat=vae_a.get("c_lat", 4), base_ch=vae_a.get("base_ch", 32)).to(device)
    vae.load_state_dict(vae_ck["model"]); vae.eval()

    ldm_ck  = torch.load(args.ldm_ckpt, map_location=device, weights_only=False)
    ldm_a   = ldm_ck.get("args", {})
    c_lat   = ldm_ck.get("c_lat", vae.c_lat)
    lat_m   = LatentUNet(in_ch=c_lat, base_ch=ldm_a.get("base_ch", 64),
                         time_dim=ldm_a.get("time_dim", 256)).to(device)
    lat_m.load_state_dict(ldm_ck["model"]); lat_m.eval()
    lat_diff = DDPM(T=ldm_a.get("T", 1000), beta_schedule=ldm_a.get("schedule", "linear"),
                    device=device, noise_std=ldm_ck.get("noise_std", 1.0))
    print(f"  Latent noise_std={lat_diff.noise_std:.5f}")

    # ── Setup results storage ─────────────────────────────────────────────
    methods = ["repaint_r1", "repaint_r10", "dps_z004", "latent_dps_z004"]
    results = {m: {"rmse": [], "mag_rmse": [], "angle_err": [], "time": []} for m in methods}
    csv_rows = {m: [f"sample_idx,rmse,mag_rmse,angle_err_deg,time_s"] for m in methods}

    n_samples = min(args.n_samples, test_fields.shape[0] - args.sample_start)
    idxs = list(range(args.sample_start, args.sample_start + n_samples))

    for c, idx in enumerate(idxs, start=1):
        true      = test_fields[idx]
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps, seed=args.seed + idx)
        x_obs     = true.copy(); x_obs[:, ~path_mask] = 0.0

        preds = {}

        # ── Repaint r=1 ───────────────────────────────────────────────────
        t0 = time.time()
        preds["repaint_r1"] = repaint_infer(base_m, base_diff, torch.from_numpy(x_obs),
                                             path_mask, land_mask, device,
                                             stride=args.stride, resample=1)
        dt_r1 = time.time() - t0

        # ── Repaint r=10 ──────────────────────────────────────────────────
        t0 = time.time()
        preds["repaint_r10"] = repaint_infer(base_m, base_diff, torch.from_numpy(x_obs),
                                              path_mask, land_mask, device,
                                              stride=args.stride, resample=10)
        dt_r10 = time.time() - t0

        # ── Pixel DPS ─────────────────────────────────────────────────────
        t0 = time.time()
        preds["dps_z004"] = dps_infer(base_m, base_diff, x_obs,
                                       path_mask, land_mask, device,
                                       stride=args.stride, zeta=args.zeta)
        dt_dps = time.time() - t0

        # ── Latent DPS ────────────────────────────────────────────────────
        t0 = time.time()
        preds["latent_dps_z004"] = latent_dps_infer(vae, lat_m, lat_diff, x_obs,
                                                     path_mask, land_mask, device,
                                                     stride=args.stride, zeta=args.zeta)
        dt_ldm = time.time() - t0

        times = {"repaint_r1": dt_r1, "repaint_r10": dt_r10,
                 "dps_z004": dt_dps, "latent_dps_z004": dt_ldm}

        for m in methods:
            pr   = preds[m]
            rmse = rmse_ocean(pr, true, ocean_mask)
            mag  = mag_rmse_ocean(pr, true, ocean_mask)
            ang  = angle_error_ocean(pr, true, ocean_mask)
            t    = times[m]
            results[m]["rmse"].append(rmse)
            results[m]["mag_rmse"].append(mag)
            results[m]["angle_err"].append(ang)
            results[m]["time"].append(t)
            csv_rows[m].append(f"{idx},{rmse:.8f},{mag:.8f},{ang:.4f},{t:.3f}")

        if c % 5 == 0 or c == n_samples:
            print(f"  [{c}/{n_samples}]  "
                  f"r1={results['repaint_r1']['rmse'][-1]:.4f}  "
                  f"r10={results['repaint_r10']['rmse'][-1]:.4f}  "
                  f"dps={results['dps_z004']['rmse'][-1]:.4f}  "
                  f"ldm={results['latent_dps_z004']['rmse'][-1]:.4f}  "
                  f"[t: {dt_r1:.1f}/{dt_r10:.1f}/{dt_dps:.1f}/{dt_ldm:.1f}s]")

        # ── Save image if requested ───────────────────────────────────────
        if idx in img_idxs:
            spd  = np.sqrt(true[0]**2 + true[1]**2)
            vmax = max(float(np.nanpercentile(spd[ocean_mask], 98)), 1e-6)

            fig, axes = plt.subplots(2, 5, figsize=(26, 10))
            fig.suptitle(
                f"Stride={args.stride} Comparison  —  sample_idx={idx}  seed={args.seed}  "
                f"({100*path_mask.sum()/ocean_mask.sum():.1f}% observed)",
                fontsize=12, fontweight="bold"
            )

            labels = {"repaint_r1": "RePaint r=1", "repaint_r10": "RePaint r=10",
                      "dps_z004": f"DPS ζ={args.zeta}", "latent_dps_z004": f"Latent DPS ζ={args.zeta}"}

            plot_field(axes[0,0], true, land_mask, "Ground Truth", vmax=vmax)
            for col, m in enumerate(methods, start=1):
                r = results[m]["rmse"][-1]
                t = results[m]["time"][-1]
                plot_field(axes[0,col], preds[m], land_mask,
                           f"{labels[m]}\nRMSE={r:.4f}  t={t:.1f}s", vmax=vmax)

            plot_path(axes[1,0], land_mask, path_mask,
                      f"Robot Path ({int(path_mask.sum())} cells)")
            for col, m in enumerate(methods, start=1):
                r = results[m]["rmse"][-1]
                plot_error(axes[1,col], preds[m], true, land_mask,
                           f"Error — {labels[m]}\nRMSE={r:.4f}",
                           add_cbar=(col == len(methods)))

            plt.tight_layout()
            out_img = os.path.join(args.out_dir, "images", f"sample_{idx:03d}.png")
            plt.savefig(out_img, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved image: {out_img}")

    # ── Summary ───────────────────────────────────────────────────────────
    lines = [
        f"Stride={args.stride} Comparison  (T=1000 → {1000//args.stride} effective steps)",
        f"n_samples={n_samples}  path_steps={args.path_steps}  seed={args.seed}  ζ={args.zeta}",
        "",
        f"{'Method':<22} {'RMSE':>10} {'±std':>8} {'MagRMSE':>10} {'AngleErr°':>11} {'Time(s)':>9}",
        "-" * 76,
    ]
    csv_summary = ["model,mean_rmse,std_rmse,mag_rmse,angle_err_deg,mean_time_s,n_samples"]

    for m in methods:
        rmse_arr = np.array(results[m]["rmse"])
        mag_arr  = np.array(results[m]["mag_rmse"])
        ang_arr  = np.array([x for x in results[m]["angle_err"] if not np.isnan(x)])
        t_arr    = np.array(results[m]["time"])
        mr, sr   = rmse_arr.mean(), rmse_arr.std()
        mm       = mag_arr.mean()
        ma       = ang_arr.mean() if len(ang_arr) else float("nan")
        mt       = t_arr.mean()
        lines.append(f"{m:<22} {mr:10.6f} {sr:8.6f} {mm:10.6f} {ma:11.2f} {mt:9.2f}")
        csv_summary.append(f"{m},{mr:.8f},{sr:.8f},{mm:.8f},{ma:.4f},{mt:.3f},{n_samples}")
        with open(os.path.join(args.out_dir, f"{m}_per_sample.csv"), "w") as f:
            f.write("\n".join(csv_rows[m]) + "\n")

    summary_str = "\n".join(lines)
    print("\n" + summary_str)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    with open(os.path.join(args.out_dir, "summary.csv"), "w") as f:
        f.write("\n".join(csv_summary) + "\n")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
