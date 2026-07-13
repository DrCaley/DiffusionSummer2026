"""
infer_fno_hetero_compare.py
==============================
Evaluates the heteroscedastic single-shot FNO (FNOHetero): for each of
N_seeds (seed, test-sample) pairs, runs one deterministic forward pass to get
(mean, log_var), then draws N_runs independent stochastic samples
x0 ~ N(mean, exp(log_var)) from the *same* mean/variance field (same
observation, same path mask — only the sampling noise differs between runs).

Reports, per seed and averaged over all seeds:
  - Mean-prediction accuracy (RMSE / mag err / angle err) — the deterministic
    baseline, i.e. what you'd get with zero sampling noise.
  - Per-run accuracy averaged over N_runs, plus its RUN-TO-RUN STANDARD
    DEVIATION — this is the actual non-determinism: how much do independent
    samples of the same (seed, observation) differ from each other.
  - Predicted sigma (the model's own learned uncertainty, averaged over ocean
    pixels) vs. the empirical std of the N_runs realizations at each pixel,
    averaged over ocean — a calibration sanity check (they should roughly
    agree if the model's uncertainty estimate is well-calibrated).

Usage:
    python3 infer_fno_hetero_compare.py \\
        --pickle /root/ocean_ddpm/data_local.pickle \\
        --checkpoint /root/NeuralOperator/checkpoints_fno_hetero/best_fno_hetero.pt \\
        --out_dir /root/NeuralOperator/results/hetero_T_50seeds_20runs \\
        --n_seeds 50 --n_runs 20 --path_steps 150
"""

import argparse
import csv
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


SEEDS = list(range(0, 700, 7))


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
    p.add_argument("--out_dir",       default="hetero_compare")
    p.add_argument("--n_seeds",       type=int, default=50)
    p.add_argument("--n_runs",        type=int, default=20)
    p.add_argument("--path_steps",    type=int, default=150)
    args = p.parse_args()

    diff_dir = _find_diffusion_dir(args.diffusion_dir)
    sys.path.insert(0, diff_dir)
    print(f"Using dataset helpers from: {diff_dir}")

    from dataset           import OceanCurrentDataset
    from repaint_infer      import biased_walk_path
    from model_fno_hetero   import FNOHetero, build_input

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_args = ckpt.get("args", {})
    model = FNOHetero(
        in_ch=4, out_ch=2, width=ck_args.get("width", 64),
        modes1=ck_args.get("modes1", 16), modes2=ck_args.get("modes2", 16),
        n_layers=ck_args.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"val_nll={ckpt.get('val_loss', float('nan')):.5f}, "
          f"val_mean_rmse={ckpt.get('val_mean_rmse', float('nan')):.5f})")

    test_ds    = OceanCurrentDataset(args.pickle, split=2)
    land_mask  = test_ds.land_mask.numpy()
    ocean_mask = ~land_mask
    ocean_t    = torch.from_numpy(ocean_mask).float()[None, None].to(device)
    n_test     = len(test_ds)

    seeds   = SEEDS[:args.n_seeds]
    n_total = len(seeds)
    print(f"Seeds ({n_total}) x {args.n_runs} runs each\n", flush=True)

    rows = []
    mean_rmses, mean_mags, mean_angs = [], [], []
    run_rmse_means, run_rmse_stds = [], []
    run_mag_means, run_mag_stds   = [], []
    run_ang_means, run_ang_stds   = [], []
    sigma_pred_means, sigma_emp_means = [], []

    with torch.no_grad():
        for run_i, seed in enumerate(seeds):
            sample_idx = seed % n_test
            x0_true    = test_ds[sample_idx]
            true_np    = x0_true.numpy()
            path_mask  = biased_walk_path(land_mask, n_steps=args.path_steps, seed=seed)
            path_t     = torch.from_numpy(path_mask).float()[None, None].to(device)
            x0_obs     = (x0_true.unsqueeze(0).to(device) * path_t)

            inp = build_input(x0_obs, path_t, ocean_t)
            mean, log_var = model(inp)                      # (1, 2, H, W) each
            sigma = (0.5 * log_var).exp()                    # (1, 2, H, W)

            mean_np = mean.squeeze(0).cpu().numpy()
            rmse_m, mag_m, ang_m = compute_errors(mean_np, true_np, ocean_mask)
            mean_rmses.append(rmse_m); mean_mags.append(mag_m); mean_angs.append(ang_m)

            sigma_pred_mean = float(sigma.squeeze(0).cpu().numpy()[:, ocean_mask].mean())
            sigma_pred_means.append(sigma_pred_mean)

            run_preds = np.zeros((args.n_runs, 2, *true_np.shape[1:]), dtype=np.float32)
            run_rmses, run_mags, run_angs = [], [], []
            for k in range(args.n_runs):
                eps_k = torch.randn_like(mean)
                sample_k = mean + sigma * eps_k
                sample_np = sample_k.squeeze(0).cpu().numpy()
                run_preds[k] = sample_np
                r_, m_, a_ = compute_errors(sample_np, true_np, ocean_mask)
                run_rmses.append(r_); run_mags.append(m_); run_angs.append(a_)

            run_rmse_mean, run_rmse_std = float(np.mean(run_rmses)), float(np.std(run_rmses))
            run_mag_mean,  run_mag_std  = float(np.mean(run_mags)),  float(np.std(run_mags))
            run_ang_mean,  run_ang_std  = float(np.mean(run_angs)),  float(np.std(run_angs))
            run_rmse_means.append(run_rmse_mean); run_rmse_stds.append(run_rmse_std)
            run_mag_means.append(run_mag_mean);   run_mag_stds.append(run_mag_std)
            run_ang_means.append(run_ang_mean);   run_ang_stds.append(run_ang_std)

            # empirical std across the N_runs realizations, per pixel, averaged over ocean
            sigma_emp_mean = float(run_preds.std(axis=0)[:, ocean_mask].mean())
            sigma_emp_means.append(sigma_emp_mean)

            print(f"[{run_i+1:02d}/{n_total:02d}] seed={seed:4d}  "
                  f"mean_pred: rmse={rmse_m:.4f}  |  "
                  f"runs({args.n_runs}): rmse={run_rmse_mean:.4f}±{run_rmse_std:.4f}  "
                  f"ang={run_ang_mean:5.1f}±{run_ang_std:.1f}°  |  "
                  f"sigma: pred={sigma_pred_mean:.4f} emp={sigma_emp_mean:.4f}", flush=True)

            rows.append([seed, sample_idx, rmse_m, mag_m, ang_m,
                        run_rmse_mean, run_rmse_std, run_mag_mean, run_mag_std,
                        run_ang_mean, run_ang_std, sigma_pred_mean, sigma_emp_mean])

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    header = ["seed", "test_idx", "mean_pred_rmse", "mean_pred_mag_err", "mean_pred_angle_err_deg",
              "run_rmse_mean", "run_rmse_std", "run_mag_mean", "run_mag_std",
              "run_angle_mean_deg", "run_angle_std_deg", "sigma_pred_mean", "sigma_empirical_mean"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"FNO Heteroscedastic Single-Shot Regression\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"N seeds    : {n_total}\n")
        f.write(f"N runs/seed: {args.n_runs}\n")
        f.write(f"Path steps : {args.path_steps}\n\n")
        f.write("Deterministic mean prediction (zero sampling noise):\n")
        f.write(f"  RMSE     : {np.mean(mean_rmses):.4f}  (std across seeds: {np.std(mean_rmses):.4f})\n")
        f.write(f"  MagErr   : {np.mean(mean_mags):.4f}\n")
        f.write(f"  AngErr   : {np.mean(mean_angs):.2f} deg\n\n")
        f.write(f"Stochastic samples ({args.n_runs} runs per seed, same obs/mask, fresh noise):\n")
        f.write(f"  RMSE     : mean={np.mean(run_rmse_means):.4f}  "
                f"avg run-to-run std={np.mean(run_rmse_stds):.4f}\n")
        f.write(f"  MagErr   : mean={np.mean(run_mag_means):.4f}  "
                f"avg run-to-run std={np.mean(run_mag_stds):.4f}\n")
        f.write(f"  AngErr   : mean={np.mean(run_ang_means):.2f} deg  "
                f"avg run-to-run std={np.mean(run_ang_stds):.2f} deg\n\n")
        f.write(f"RMSE inflation from sampling (run mean - deterministic mean): "
                f"{np.mean(run_rmse_means) - np.mean(mean_rmses):+.4f}\n\n")
        f.write(f"Sigma calibration (avg over ocean pixels):\n")
        f.write(f"  Predicted sigma (model's own uncertainty) : {np.mean(sigma_pred_means):.4f}\n")
        f.write(f"  Empirical std of the {args.n_runs} realizations : {np.mean(sigma_emp_means):.4f}\n\n")
        f.write("Per-seed breakdown:\n")
        f.write(f"  {'Seed':>6} {'MeanRMSE':>9} {'RunRMSE':>9} {'RunStd':>8} "
                f"{'RunAng':>8} {'AngStd':>7} {'SigPred':>8} {'SigEmp':>7}\n")
        f.write("-" * 72 + "\n")
        for row in rows:
            seed, idx, rmse_m, mag_m, ang_m, rr_m, rr_s, rm_m, rm_s, ra_m, ra_s, sp, se = row
            f.write(f"  {seed:6d} {rmse_m:9.4f} {rr_m:9.4f} {rr_s:8.4f} "
                    f"{ra_m:8.1f} {ra_s:7.1f} {sp:8.4f} {se:7.4f}\n")

    print(f"\nCSV saved     : {csv_path}")
    print(f"Summary saved : {summary_path}")

    # ── Bar chart: deterministic vs stochastic, plus sigma calibration
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"FNO Heteroscedastic Single-Shot  —  {n_total} seeds x {args.n_runs} runs", fontsize=11)

    ax = axes[0]
    vals = [np.mean(mean_rmses), np.mean(run_rmse_means)]
    errs = [np.std(mean_rmses),  np.mean(run_rmse_stds)]
    bars = ax.bar(["Deterministic\n(mean)", "Stochastic\n(sampled)"], vals, yerr=errs,
                  capsize=5, color=["#4C72B0", "#C44E52"], alpha=0.85)
    ax.set_ylabel("RMSE"); ax.set_title("RMSE: deterministic vs sampled")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    ax.bar(["Predicted σ", "Empirical σ"], [np.mean(sigma_pred_means), np.mean(sigma_emp_means)],
          color=["#55A868", "#8172B2"], alpha=0.85)
    ax.set_ylabel("σ (speed units)"); ax.set_title("Uncertainty calibration")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)

    ax = axes[2]
    ax.hist(run_rmse_stds, bins=20, color="#DD8452", alpha=0.85)
    ax.set_xlabel("Run-to-run RMSE std (per seed)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of run-to-run variability")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)

    plt.tight_layout()
    chart_path = os.path.join(args.out_dir, "bar_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved: {chart_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
