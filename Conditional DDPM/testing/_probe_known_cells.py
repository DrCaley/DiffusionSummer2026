"""
Quick diagnostic: how accurately does the current model REPRODUCE the known
(robot-path) cells it is handed as conditioning?

The model gets the true (u,v) at the path cells via the observation channels
(obs_u, obs_v) + path_mask. This probe samples an ensemble and measures, at the
KNOWN cells only, how far the sampled field is from the ground-truth values that
were fed in -- per ensemble member and for the ensemble mean.

Reported per frame and in aggregate:
  rmse_known      RMSE over known cells (std-normalized units the model works in)
  rmse_known_pct  that RMSE as a % of the RMS magnitude of the true known values
  cos_known       mean cosine similarity (direction) at known cells
  ang_known       mean angular error (deg) at known cells
For reference, the SAME numbers over UNOBSERVED cells (dist>2) are printed so the
known-cell fidelity can be read against the hard far-field.

Run locally (MPS) so it doesn't disturb the servers:
  .venv/bin/python "Conditional DDPM/testing/_probe_known_cells.py" \
      --checkpoint Models/StreamFn_Cond_x0_mag.pt \
      --pickle Datasets/pickles/data_divfree_chrono.pickle \
      --split 2 --n_frames 8 --seed 0 --n_ensemble 8 --inference_steps 100
"""
import os, sys, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "..", "utils"),
           os.path.join(_HERE, "..", "..", "DDPM", "model"),
           os.path.join(_HERE, "..", "..")):
    sys.path.insert(0, os.path.abspath(_p))

import infer_cond as IC
from infer_cond import (
    ConditionalOceanDataset, StreamFunctionUNet, DDPM,
    build_cond, ensemble_infer, distance_to_path, cond_channels,
)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def angle_stats(pred, true, mask):
    """Mean cosine + angular error (deg) over masked cells. pred/true (2,H,W)."""
    pu, pv = pred[0][mask], pred[1][mask]
    tu, tv = true[0][mask], true[1][mask]
    pn = np.sqrt(pu * pu + pv * pv) + 1e-8
    tn = np.sqrt(tu * tu + tv * tv) + 1e-8
    cos = (pu * tu + pv * tv) / (pn * tn)
    cos = np.clip(cos, -1.0, 1.0)
    return float(cos.mean()), float(np.degrees(np.arccos(cos)).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pickle", required=True)
    p.add_argument("--split", type=int, default=2)
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ensemble", type=int, default=8)
    p.add_argument("--inference_steps", type=int, default=100)
    p.add_argument("--path_steps", type=int, default=90)
    args = p.parse_args()
    args.capture_every = 10_000  # we don't need trajectory frames

    device = select_device()
    print(f"Device     : {device}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pred_type = ckpt.get("pred_type")
    args.pred_type = pred_type
    ca        = ckpt.get("args", {})
    base_ch   = ca.get("base_ch", 64)
    time_dim  = ca.get("time_dim", 256)
    T         = ca.get("T", 1000)
    noise_type = ca.get("noise_type", "div_free")
    schedule  = ca.get("schedule", "cosine")
    lags      = tuple(ckpt.get("lags", ca.get("lags", (13, 25))))
    cond_ch   = ckpt.get("cond_ch", cond_channels(lags))
    data_mean = ckpt.get("data_mean", 0.0)
    data_std  = ckpt.get("data_std", None)
    spectral_filter = ckpt.get("spectral_filter", None)
    print(f"Model      : {os.path.basename(args.checkpoint)} ep={ckpt.get('epoch','?')} "
          f"pred={pred_type} cond_ch={cond_ch} data_std={data_std}")

    ds = ConditionalOceanDataset(
        args.pickle, split=args.split, lags=lags,
        data_mean=data_mean, data_std=data_std,
        path_steps=args.path_steps, deterministic=True,
    )
    land_np  = ds.land_mask.cpu().numpy().astype(bool)
    ocean_np = ~land_np
    n_ocean  = int(ocean_np.sum())

    stream_model = StreamFunctionUNet(
        in_ch=2, base_ch=base_ch, time_dim=time_dim, cond_ch=cond_ch).to(device)
    stream_model.load_state_dict(ckpt["model"])
    stream_model.eval()
    diffusion = DDPM(T=T, beta_schedule=schedule, device=device,
                     noise_type=noise_type, spectral_filter=spectral_filter)

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(ds), size=args.n_frames).tolist()
    print(f"split={args.split} n_valid={len(ds)} ocean={n_ocean} "
          f"frames={args.n_frames} n_ensemble={args.n_ensemble} steps={args.inference_steps}\n")

    hdr = (f"{'frame':>6} {'%kn':>5} {'rmseK':>7} {'rmseK%':>7} {'cosK':>6} "
           f"{'angK':>6} | {'rmseU':>7} {'cosU':>6} {'angU':>6}")
    print(hdr)
    rows = []
    for idx in indices:
        b = build_cond(ds, idx, args.path_steps, seed=idx)
        true_np   = b["target"].cpu().numpy()                    # (2,H,W) std-units
        path_mask = b["path_mask"]                               # (H,W) bool
        known = path_mask & ocean_np
        cov_pct = 100.0 * known.sum() / n_ocean

        mean_np, _f0, members = ensemble_infer(
            stream_model, diffusion, b["cond"], land_np, args, device, base_seed=idx)

        dist  = distance_to_path(path_mask, ocean_np)
        unobs = (dist > 2.0) & ocean_np

        # known-cell true RMS magnitude (for % normalization)
        true_rms_known = float(np.sqrt(np.mean(true_np[:, known] ** 2))) + 1e-8

        # ensemble-mean reproduction at known cells
        rmseK = float(np.sqrt(np.mean((mean_np[:, known] - true_np[:, known]) ** 2)))
        cosK, angK = angle_stats(mean_np, true_np, known)
        rmseU = float(np.sqrt(np.mean((mean_np[:, unobs] - true_np[:, unobs]) ** 2)))
        cosU, angU = angle_stats(mean_np, true_np, unobs)
        rmseK_pct = 100.0 * rmseK / true_rms_known

        rows.append((rmseK, rmseK_pct, cosK, angK, rmseU, cosU, angU))
        print(f"{int(ds.valid[idx]):>6} {cov_pct:4.1f}% {rmseK:7.4f} {rmseK_pct:6.1f}% "
              f"{cosK:+.3f} {angK:6.1f} | {rmseU:7.4f} {cosU:+.3f} {angU:6.1f}")

    a = np.array(rows)
    print("\n  MEAN over frames:")
    print(f"    KNOWN  cells : rmse={a[:,0].mean():.4f}  (={a[:,1].mean():.1f}% of true mag)  "
          f"cos={a[:,2].mean():+.3f}  angle={a[:,3].mean():.1f}deg")
    print(f"    UNOBS  cells : rmse={a[:,4].mean():.4f}  "
          f"cos={a[:,5].mean():+.3f}  angle={a[:,6].mean():.1f}deg")
    print("\n  Interpretation: rmseK% near 0 = model copies the known values well.")
    print("  If rmseK% is large, the soft conditioning is being under-used at known cells.")


if __name__ == "__main__":
    main()
