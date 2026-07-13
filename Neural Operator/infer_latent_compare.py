"""
infer_latent_compare.py
=========================
Stage 3 of FNO latent diffusion: RePaint/DPS inference done entirely in
latent space, then decoded back to pixel space for evaluation.

Path observations live in pixel space, so conditioning is necessarily
approximate here (unlike pixel-space RePaint, which merges exact known
u/v values): the sparse observed field x0_obs (true values at path cells,
zero elsewhere) is encoded through the *same* frozen encoder used for
training to get an approximate "known latent" z0_known, and the pixel-space
path mask is max-pooled down to latent resolution (any path cell inside a
4x4 block marks that latent cell as "known"). The reverse-diffusion merge
step is otherwise identical in structure to repaint_infer.py's pixel-space
version, just operating on latent tensors with no land/ocean masking (there
is no clean per-latent-pixel land split after the encoder's downsampling).

Reports the same 3 metrics as infer_compare_unet_fno.py (RMSE, magnitude
error, angle error), decoded back to pixel space and evaluated on true ocean
pixels — directly comparable to the UNet / pixel-space FNO-DDPM results.

Two DPS variants are compared:
  - "DPS z=0.04 (latent)"  — the original shortcut: measures the residual
    entirely in latent coordinates, (z0_hat - z0_known)^2, and never touches
    the decoder during the sampling loop. z0_known = encoder(x0_obs) is
    itself an approximation (the encoder never saw sparse/masked fields
    during training), and "close in latent space" isn't the same thing as
    "correct at the known pixels."
  - "DPS z=0.04 (decode)"  — the theoretically correct posterior-sampling
    formulation (PSLD / ReSample-style): at each step, decode z0_hat all the
    way to pixel space, x0_hat = decoder(z0_hat), and measure the residual
    where the measurement operator is actually defined — path_mask ⊙
    (x0_hat - x0_known_pixels). The gradient is backpropagated through the
    (frozen-weight, but graph-attached) decoder and the denoiser back to zt.
    Costs one extra decoder forward+backward per step (cheap, ~2.4M params).

Usage:
    python3 infer_latent_compare.py \\
        --pickle /root/ocean_ddpm/data_local.pickle \\
        --ae_checkpoint /root/NeuralOperator/checkpoints_fno_autoencoder/best_fno_autoencoder.pt \\
        --latent_checkpoint /root/NeuralOperator/checkpoints_latent_fno_ddpm/best_latent_fno_ddpm_linear.pt \\
        --out_dir /root/NeuralOperator/results/latent_compare_T1000_s10_50seeds \\
        --n_seeds 50 --T 1000 --stride 10 --path_steps 150
"""

import argparse
import csv
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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
        if os.path.isfile(os.path.join(d, "diffusion.py")):
            return d
    raise RuntimeError(f"Cannot find diffusion.py — tried: {candidates}")


METHODS = ["RePaint r=10", "RePaint r=1", "DPS z=0.04 (latent)", "DPS z=0.04 (decode)"]

# Total downsample factor from pixel (padded 96x44) to latent (24x11): 4x.
_MASK_DOWNSAMPLE = 4
_PAD = (0, 0, 1, 1)  # matches model_fno_autoencoder._PAD


def downsample_mask(path_mask_np: np.ndarray, device: str) -> torch.Tensor:
    """(94, 44) bool -> (1, 1, latent_H, latent_W) float, 1 = known."""
    t = torch.from_numpy(path_mask_np).float()[None, None].to(device)
    t = F.pad(t, _PAD)
    t = F.max_pool2d(t, kernel_size=_MASK_DOWNSAMPLE, stride=_MASK_DOWNSAMPLE)
    return (t > 0).float()


@torch.no_grad()
def latent_repaint_infer(model, diffusion, z0_known, latent_mask,
                         r=1, device="cpu", stride=1):
    """z0_known, latent_mask: (1, C, h, w). Returns (1, C, h, w)."""
    _, C, h, w = z0_known.shape
    zt = torch.randn(1, C, h, w, device=device) * diffusion.noise_std
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0
        for j in range(r):
            zt_unk = diffusion.p_sample_step(model, zt, t_int, t_prev_int)
            t_prev = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            zt_kn, _ = diffusion.q_sample(z0_known, t_prev)
            zt = latent_mask * zt_kn + (1 - latent_mask) * zt_unk
            if j < r - 1 and t_int > 0:
                zt = diffusion.q_sample_from_prev(zt, t_int, t_prev_int)

    return zt


def latent_dps_infer(model, diffusion, z0_known, latent_mask,
                     device="cpu", stride=1, step_size=0.04):
    """Shortcut DPS: residual measured entirely in latent coordinates."""
    _, C, h, w = z0_known.shape
    zt = torch.randn(1, C, h, w, device=device) * diffusion.noise_std
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        zt_in = zt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        eps = model(zt_in, t_vec)
        ab  = diffusion.alpha_bar[t_int]
        z0h = (zt_in - (1 - ab).sqrt() * eps) / ab.sqrt()
        nsq = (latent_mask * (z0h - z0_known) ** 2).sum()
        grad = torch.autograd.grad(nsq, zt_in)[0]

        with torch.no_grad():
            zt = diffusion.p_sample_step(model, zt_in.detach(), t_int, t_prev_int)
            zt = zt - (step_size / (nsq.sqrt().item() + 1e-8)) * grad.detach()

    return zt.detach()


def latent_dps_infer_decode(model, diffusion, decoder, x0_known_pixel, path_mask_pixel,
                            latent_shape, device="cpu", stride=1, step_size=0.04):
    """
    Decode-through DPS (PSLD / ReSample-style): the measurement residual is
    computed in pixel space — where the path mask is actually well-defined —
    by decoding z0_hat through the (frozen-weight, graph-attached) decoder
    before comparing against the true known pixels. Gradient flows through
    the decoder and the denoiser back to zt.

    x0_known_pixel, path_mask_pixel: (1, 2, 94, 44) / (1, 1, 94, 44).
    latent_shape: (C, h, w) of the latent tensor.
    """
    C, h, w = latent_shape
    zt = torch.randn(1, C, h, w, device=device) * diffusion.noise_std
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        zt_in = zt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        eps = model(zt_in, t_vec)
        ab  = diffusion.alpha_bar[t_int]
        z0h = (zt_in - (1 - ab).sqrt() * eps) / ab.sqrt()

        x0h = decoder(z0h)                              # decode to pixel space
        residual = path_mask_pixel * (x0h - x0_known_pixel)
        nsq = (residual ** 2).sum()
        grad = torch.autograd.grad(nsq, zt_in)[0]

        with torch.no_grad():
            zt = diffusion.p_sample_step(model, zt_in.detach(), t_int, t_prev_int)
            zt = zt - (step_size / (nsq.sqrt().item() + 1e-8)) * grad.detach()

    return zt.detach()


def compute_errors(pred, truth, ocean_mask, eps=1e-8):
    pu, pv = pred[0][ocean_mask],  pred[1][ocean_mask]
    tu, tv = truth[0][ocean_mask], truth[1][ocean_mask]

    rmse = float(np.sqrt(np.mean((pu - tu)**2 + (pv - tv)**2)))

    pred_mag = np.sqrt(pu**2 + pv**2)
    true_mag = np.sqrt(tu**2 + tv**2)
    mag_err  = float(np.mean(np.abs(pred_mag - true_mag)))

    dot = pu * tu + pv * tv
    cos = np.clip(dot / (pred_mag * true_mag + eps), -1.0, 1.0)
    angle_err_deg = float(np.degrees(np.mean(np.arccos(cos))))

    return rmse, mag_err, angle_err_deg


def save_bar_chart(all_rmse, all_mag, all_ang, all_times, T, stride, n_seeds, out_path):
    keys  = list(all_rmse.keys())
    rmse  = [np.mean(all_rmse[k]) for k in keys]
    rstd  = [np.std(all_rmse[k])  for k in keys]
    mag   = [np.mean(all_mag[k])  for k in keys]
    mstd  = [np.std(all_mag[k])   for k in keys]
    ang   = [np.mean(all_ang[k])  for k in keys]
    astd  = [np.std(all_ang[k])   for k in keys]
    times = [np.mean(all_times[k]) for k in keys]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    x = np.arange(len(keys))
    w = 0.6

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.suptitle(f"Latent FNO-DDPM (RePaint/DPS in latent space)  —  T={T} / stride={stride}  —  {n_seeds} seeds",
                fontsize=11)

    def _bar(ax, vals, errs, title, ylabel, fmt):
        bars = ax.bar(x, vals, w, yerr=errs, capsize=4, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=7, rotation=15, ha="right")
        ax.set_ylim(0, max(vals) * 1.5 + 1e-9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=8)

    _bar(axes[0], rmse,  rstd, "Mean RMSE (± 1 std)",            "RMSE",      "{:.4f}")
    _bar(axes[1], mag,   mstd, "Mean Magnitude Error (± 1 std)", "|Δ speed|", "{:.4f}")
    _bar(axes[2], ang,   astd, "Mean Angle Error (± 1 std)",     "Degrees",   "{:.1f}°")
    _bar(axes[3], times, None, "Mean Inference Time / seed",     "Seconds",   "{:.1f}s")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",            default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--ae_checkpoint",     required=True)
    p.add_argument("--latent_checkpoint", required=True)
    p.add_argument("--diffusion_dir",     default=None)
    p.add_argument("--out_dir",           default="latent_compare")
    p.add_argument("--n_seeds",           type=int, default=50)
    p.add_argument("--T",                 type=int, default=1000)
    p.add_argument("--stride",            type=int, default=10)
    p.add_argument("--path_steps",        type=int, default=150)
    args = p.parse_args()

    diff_dir = _find_diffusion_dir(args.diffusion_dir)
    sys.path.insert(0, diff_dir)
    print(f"Using dataset/diffusion helpers from: {diff_dir}")

    from dataset               import OceanCurrentDataset
    from diffusion             import DDPM
    from repaint_infer         import biased_walk_path
    from model_fno_ddpm        import FNO2dDDPM
    from model_fno_autoencoder import FNOAutoencoder

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    SEEDS = list(range(0, 700, 7))

    # ---- Load autoencoder ----
    ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    ae_args = ae_ckpt.get("args", {})
    autoencoder = FNOAutoencoder(
        in_ch=2, base=ae_args.get("base", 32), latent_ch=ae_args.get("latent_ch", 8),
        modes1=ae_args.get("modes1", 12), modes2=ae_args.get("modes2", 6),
        n_blocks=ae_args.get("n_blocks", 2),
    ).to(device)
    autoencoder.load_state_dict(ae_ckpt["model"])
    autoencoder.eval()
    for p_ in autoencoder.parameters():
        p_.requires_grad_(False)   # frozen weights; decoder activations still need
                                    # to be graph-attached for the decode-through DPS gradient
    print(f"Loaded autoencoder (epoch {ae_ckpt.get('epoch','?')}, val={ae_ckpt.get('val_loss', float('nan')):.6f})")

    # ---- Load latent DDPM ----
    lat_ckpt = torch.load(args.latent_checkpoint, map_location="cpu", weights_only=False)
    lat_args = lat_ckpt.get("args", {})
    model = FNO2dDDPM(
        in_ch=lat_ckpt["latent_ch"], width=lat_args.get("width", 64),
        modes1=lat_args.get("modes1", 12), modes2=lat_args.get("modes2", 6),
        time_dim=lat_args.get("time_dim", 256), n_layers=lat_args.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(lat_ckpt["model"])
    model.eval()
    schedule  = lat_ckpt.get("schedule", "linear")
    noise_std = lat_ckpt.get("noise_std", 1.0)
    print(f"Loaded latent FNO-DDPM (epoch {lat_ckpt.get('epoch','?')}, schedule={schedule}, "
          f"noise_std={noise_std:.5f})")

    diffusion = DDPM(T=args.T, beta_schedule=schedule, device=device,
                     noise_std=noise_std, curl_div_weight=0.0)

    # ---- Data ----
    test_ds    = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = test_ds.land_mask.numpy()
    ocean_mask = ~land_mask
    n_test     = len(test_ds)

    all_rmse  = {m: [] for m in METHODS}
    all_mag   = {m: [] for m in METHODS}
    all_ang   = {m: [] for m in METHODS}
    all_times = {m: [] for m in METHODS}
    rows = []

    seeds = SEEDS[:args.n_seeds]
    n_total = len(seeds)
    print(f"Seeds ({n_total}): {seeds}\n", flush=True)

    for run_i, seed in enumerate(seeds):
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]
        true_np    = x0_true.numpy()
        path_mask  = biased_walk_path(land_mask, n_steps=args.path_steps, seed=seed)
        x0_obs     = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        with torch.no_grad():
            z0_known = autoencoder.encoder(x0_obs.unsqueeze(0).to(device))
        latent_mask = downsample_mask(path_mask, device)
        latent_shape = z0_known.shape[1:]

        x0_known_pixel  = x0_obs.unsqueeze(0).to(device)                                    # (1,2,94,44)
        path_mask_pixel = torch.from_numpy(path_mask).float()[None, None].to(device)        # (1,1,94,44)

        print(f"[{run_i+1:02d}/{n_total:02d}] seed={seed}  sample={sample_idx}  "
              f"path_cells={int(path_mask.sum())}", flush=True)
        row = [seed, sample_idx]

        def run(method_name, fn, *fn_args, **fn_kwargs):
            t0 = time.perf_counter()
            zt = fn(*fn_args, **fn_kwargs)
            with torch.no_grad():
                pred = autoencoder.decoder(zt).squeeze(0).cpu().numpy()
            t = time.perf_counter() - t0
            rmse, mag_err, ang_err = compute_errors(pred, true_np, ocean_mask)
            all_rmse[method_name].append(rmse)
            all_mag[method_name].append(mag_err)
            all_ang[method_name].append(ang_err)
            all_times[method_name].append(t)
            print(f"  {method_name:<20}: rmse={rmse:.4f}  mag_err={mag_err:.4f}  "
                  f"angle_err={ang_err:6.2f}°  t={t:.1f}s", flush=True)
            return rmse, mag_err, ang_err, t

        r_, m_, a_, t_ = run("RePaint r=10", latent_repaint_infer, model, diffusion,
                             z0_known, latent_mask, r=10, device=device, stride=args.stride)
        row += [r_, m_, a_, t_]

        r_, m_, a_, t_ = run("RePaint r=1", latent_repaint_infer, model, diffusion,
                             z0_known, latent_mask, r=1, device=device, stride=args.stride)
        row += [r_, m_, a_, t_]

        r_, m_, a_, t_ = run("DPS z=0.04 (latent)", latent_dps_infer, model, diffusion,
                             z0_known, latent_mask, device=device, stride=args.stride,
                             step_size=0.04)
        row += [r_, m_, a_, t_]

        r_, m_, a_, t_ = run("DPS z=0.04 (decode)", latent_dps_infer_decode, model, diffusion,
                             autoencoder.decoder, x0_known_pixel, path_mask_pixel, latent_shape,
                             device=device, stride=args.stride, step_size=0.04)
        row += [r_, m_, a_, t_]

        rows.append(row)

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    header = ["seed", "test_idx"]
    for m in METHODS:
        key = m.replace(" ", "_").replace("=", "")
        header += [f"{key}_rmse", f"{key}_mag_err", f"{key}_angle_err_deg", f"{key}_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Latent FNO-DDPM Comparison  —  T={args.T}  stride={args.stride}\n")
        f.write(f"Autoencoder checkpoint : {args.ae_checkpoint}\n")
        f.write(f"Latent DDPM checkpoint : {args.latent_checkpoint}\n")
        f.write(f"N seeds                : {n_total}\n")
        f.write(f"Path steps             : {args.path_steps}\n\n")
        f.write(f"{'Method':<20} {'Mean RMSE':>10} {'Std RMSE':>9} "
                f"{'Mean MagErr':>12} {'Mean AngErr(deg)':>17} {'Mean Time(s)':>13}\n")
        f.write("-" * 84 + "\n")
        for m in METHODS:
            rs, ms, as_, ts = all_rmse[m], all_mag[m], all_ang[m], all_times[m]
            f.write(f"{m:<20} {np.mean(rs):>10.4f} {np.std(rs):>9.4f} "
                    f"{np.mean(ms):>12.4f} {np.mean(as_):>17.2f} {np.mean(ts):>13.2f}\n")

    print(f"\nCSV saved     : {csv_path}")
    print(f"Summary saved : {summary_path}")

    chart_path = os.path.join(args.out_dir, "bar_chart.png")
    save_bar_chart(all_rmse, all_mag, all_ang, all_times, args.T, args.stride, n_total, chart_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
