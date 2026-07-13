"""
batch_rmse_ensemble_ae.py
Ensemble method 4: noisy AE inputs.

For each test sample, generate K independent AE predictions by perturbing
the observed path cells: x_noisy = x_obs + σ * randn * path_mask

Reports:
  - mean RMSE of individual ensemble members  (accuracy)
  - std RMSE within ensemble                  (stochasticity measure)
  - RMSE of the ensemble-mean prediction      (ensemble benefit)
  - plain AE RMSE for comparison              (deterministic baseline)

Sweeps σ over [0.01, 0.05, 0.1, 0.2] with K=10 members.
"""
import argparse
import os
import pickle
import numpy as np
import torch

from ae_model import RepaintAutoencoder


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


def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask])**2)))


def ae_predict(ae_model, x_obs, path_mask, device, sigma=0.0, rng=None):
    """Run AE with optional Gaussian noise on observed cells."""
    x_in = x_obs.copy()
    if sigma > 0 and rng is not None:
        noise = rng.standard_normal(x_in[:, path_mask].shape).astype(np.float32)
        x_in[:, path_mask] += sigma * noise
    mask_ch = path_mask.astype(np.float32)[None]
    ae_inp  = torch.from_numpy(np.concatenate([x_in, mask_ch], axis=0)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = ae_model(ae_inp).squeeze(0).cpu().numpy()
    return pred


def load_split(data, split_name):
    IDX = {"train": 0, "val": 1, "test": 2}
    key = split_name if (isinstance(data, dict) and split_name in data) else IDX[split_name]
    arr = np.asarray(data[key], dtype=np.float32)
    return np.nan_to_num(np.transpose(arr, (3,2,0,1)).astype(np.float32)), np.isnan(arr[:,:,0,0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",    default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--ae_ckpt",   default="/root/autoencoder_train/checkpoints/best_model_autoencoder.pt")
    p.add_argument("--out_dir",   default="/root/autoencoder_train/inference_results")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--path_steps",   type=int, default=150)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--K",        type=int, default=10,
                   help="Ensemble size (number of noisy AE runs per sample)")
    p.add_argument("--sigmas",   default="0.0,0.01,0.05,0.1,0.2",
                   help="Comma-separated σ values to sweep (0.0 = plain AE)")
    p.add_argument("--device",   default=None)
    args = p.parse_args()

    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sigmas  = [float(x) for x in args.sigmas.split(",")]
    print(f"Device : {device}")
    print(f"K      : {args.K}  (ensemble members per sample)")
    print(f"sigmas : {sigmas}")

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    _, land_mask = load_split(data, "train")
    test_fields, land_mask = load_split(data, "test")
    ocean_mask = ~land_mask

    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae_m  = RepaintAutoencoder(in_ch=3, out_ch=2,
                                base_ch=ae_ck.get("args", {}).get("base_ch", 64)).to(device)
    ae_m.load_state_dict(ae_ck["model"]); ae_m.eval()
    print("AE loaded.")

    n_samples = min(args.n_samples, test_fields.shape[0] - args.sample_start)
    idxs = list(range(args.sample_start, args.sample_start + n_samples))

    # results[sigma] = {
    #   "member_rmse_mean": [],   # mean RMSE across K members per sample
    #   "member_rmse_std":  [],   # std RMSE across K members (stochasticity)
    #   "ensemble_rmse":    [],   # RMSE of mean prediction
    # }
    results = {s: {"member_mean": [], "member_std": [], "ensemble": []} for s in sigmas}
    rng = np.random.default_rng(args.seed + 9999)

    for c, idx in enumerate(idxs, start=1):
        true      = test_fields[idx]
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps,
                                     seed=args.seed + idx)
        x_obs = true.copy(); x_obs[:, ~path_mask] = 0.0

        for sigma in sigmas:
            K_eff = 1 if sigma == 0.0 else args.K
            preds = [ae_predict(ae_m, x_obs, path_mask, device, sigma=sigma, rng=rng)
                     for _ in range(K_eff)]

            rmses = [rmse_ocean(pr, true, ocean_mask) for pr in preds]
            mean_pred = np.mean(preds, axis=0)
            ens_rmse  = rmse_ocean(mean_pred, true, ocean_mask)

            results[sigma]["member_mean"].append(float(np.mean(rmses)))
            results[sigma]["member_std"].append(float(np.std(rmses)))
            results[sigma]["ensemble"].append(ens_rmse)

        if c % 5 == 0 or c == n_samples:
            print(f"  [{c}/{n_samples}]", end="")
            for sigma in sigmas:
                m = results[sigma]["member_mean"][-1]
                s = results[sigma]["member_std"][-1]
                print(f"  σ={sigma}: rmse={m:.4f} ±{s:.4f}", end="")
            print()

    # ── Summary ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    lines = [
        f"Ensemble AE (noisy observations) — K={args.K} members",
        f"n_samples={n_samples}  path_steps={args.path_steps}  seed={args.seed}",
        "",
        f"{'σ':<8} {'Mean RMSE':>12} {'Std RMSE':>12} {'WithinStd':>12} {'EnsRMSE':>12}",
        "-" * 62,
    ]
    csv_rows = ["sigma,mean_member_rmse,std_member_rmse,mean_within_std,ensemble_rmse,n_samples"]

    for sigma in sigmas:
        mm   = np.mean(results[sigma]["member_mean"])
        sm   = np.std(results[sigma]["member_mean"])
        ws   = np.mean(results[sigma]["member_std"])   # avg within-ensemble std → stochasticity
        em   = np.mean(results[sigma]["ensemble"])
        lines.append(f"{sigma:<8.3f} {mm:12.6f} {sm:12.6f} {ws:12.6f} {em:12.6f}")
        csv_rows.append(f"{sigma},{mm:.8f},{sm:.8f},{ws:.8f},{em:.8f},{n_samples}")

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(os.path.join(args.out_dir, "ensemble_ae_summary.txt"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.out_dir, "ensemble_ae_summary.csv"), "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"\nSaved to {args.out_dir}/ensemble_ae_summary.txt")


if __name__ == "__main__":
    main()
