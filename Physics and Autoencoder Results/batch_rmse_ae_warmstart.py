"""
batch_rmse_ae_warmstart.py
AE warm-start diffusion: run autoencoder first, forward-diffuse its output
to t_start, then run RePaint reverse from t_start (not full T=1000).

No training required — uses existing AE and diffusion checkpoints.

For each sample:
  1. AE forward pass   → x̂₀_ae
  2. q_sample(x̂₀_ae, t_start) → x_{t_start}  (add noise)
  3. RePaint r=1 reverse  from t_start → 0  (t_start steps, not 1000)

Evaluated variants:
  - ae_warmstart_base_t{t_start}    (baseline DDPM)
  - ae_warmstart_physics_t{t_start} (physics DDPM)
"""
import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

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


# ── AE warm-start RePaint ───────────────────────────────────────────────────

@torch.no_grad()
def ae_warmstart_repaint(ae_model, diff_model, diffusion,
                          x0_known_np, path_mask, land_mask,
                          device, t_start):
    """
    1. AE: (masked_field + mask_ch) → x̂₀_ae
    2. Forward diffuse x̂₀_ae to t_start
    3. RePaint r=1 reverse from t_start → 0
    """
    H, W = land_mask.shape
    x0_known = torch.from_numpy(x0_known_np).unsqueeze(0).to(device)  # (1,2,H,W)
    known_t  = torch.from_numpy(path_mask).float().to(device)[None, None]
    ocean_t  = 1.0 - torch.from_numpy(land_mask).float().to(device)[None, None]

    # ── Step 1: AE reconstruction ──────────────────────────────────────────
    x_obs    = x0_known_np.copy(); x_obs[:, ~path_mask] = 0.0
    mask_ch  = path_mask.astype(np.float32)[None]                      # (1,H,W)
    ae_inp   = np.concatenate([x_obs, mask_ch], axis=0)               # (3,H,W)
    ae_inp_t = torch.from_numpy(ae_inp).unsqueeze(0).to(device)        # (1,3,H,W)
    ae_model.eval(); diff_model.eval()
    x0_ae = ae_model(ae_inp_t)                                         # (1,2,H,W)
    x0_ae = x0_ae * ocean_t                                            # zero land

    # ── Step 2: Forward diffuse to t_start ─────────────────────────────────
    t_tensor = torch.full((1,), t_start, device=device, dtype=torch.long)
    xt, _    = diffusion.q_sample(x0_ae, t_tensor)
    xt       = xt * ocean_t

    # ── Step 3: RePaint reverse from t_start → 0 ───────────────────────────
    for t_int in reversed(range(t_start)):
        t_prev     = max(t_int - 1, 0)
        xt_unknown = diffusion.p_sample_step(diff_model, xt, t_int, t_prev)
        tp         = torch.full((1,), t_prev, device=device, dtype=torch.long)
        xt_known, _= diffusion.q_sample(x0_known, tp)
        xt         = known_t * xt_known + (1.0 - known_t) * xt_unknown
        xt         = xt * ocean_t

    return xt.squeeze(0).cpu().numpy()


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
    p.add_argument("--pickle",       default="/root/ocean_ddpm/data.pickle")
    p.add_argument("--ae_ckpt",      default="/root/autoencoder_train/checkpoints/best_model_autoencoder.pt")
    p.add_argument("--base_ckpt",    default="/root/autoencoder_train/checkpoints_linear/best_model_linear.pt")
    p.add_argument("--physics_ckpt", default="/root/autoencoder_train/checkpoints_physics/best_model_physics.pt")
    p.add_argument("--out_dir",      default="/root/autoencoder_train/inference_results")
    p.add_argument("--n_samples",    type=int,   default=50)
    p.add_argument("--sample_start", type=int,   default=0)
    p.add_argument("--path_steps",   type=int,   default=150)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--t_starts",     default="333,500",
                   help="Comma-separated t_start values to sweep")
    p.add_argument("--device",       default=None)
    args = p.parse_args()

    device    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_starts  = [int(x) for x in args.t_starts.split(",")]
    print(f"Device   : {device}")
    print(f"t_starts : {t_starts}")

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    train_fields, _         = load_split(data, "train")
    test_fields,  land_mask = load_split(data, "test")
    ocean_mask = ~land_mask

    # ── Load AE ──────────────────────────────────────────────────────────────
    ae_ck   = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae_args = ae_ck.get("args", {})
    ae_m    = RepaintAutoencoder(in_ch=3, out_ch=2,
                                  base_ch=ae_args.get("base_ch", 64)).to(device)
    ae_m.load_state_dict(ae_ck["model"]); ae_m.eval()
    print(f"AE loaded: {args.ae_ckpt}")

    # ── Load diffusion models ─────────────────────────────────────────────────
    def load_ddpm(ckpt_path, name):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        ca = ck.get("args", {})
        ns = ck.get("noise_std") or float(train_fields[:, :, ocean_mask].std())
        m  = Repaint(in_ch=2, base_ch=ca.get("base_ch", 64),
                     time_dim=ca.get("time_dim", 256)).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        d  = DDPM(T=ca.get("T", 1000), beta_schedule=ck.get("schedule", "linear"),
                  device=device, noise_std=ns)
        print(f"{name} loaded: noise_std={ns:.5f}")
        return m, d

    base_m,  base_d  = load_ddpm(args.base_ckpt,    "Base")
    phys_m,  phys_d  = load_ddpm(args.physics_ckpt, "Physics")

    n_total   = test_fields.shape[0]
    n_samples = min(args.n_samples, n_total - args.sample_start)
    idxs      = list(range(args.sample_start, args.sample_start + n_samples))

    # methods: (label, diff_model, diffusion)
    variants = [(f"ae_warmstart_base_t{{t}}", base_m, base_d),
                (f"ae_warmstart_physics_t{{t}}", phys_m, phys_d)]

    # Results dict: key = method_name, value = list of RMSEs
    results = {}
    csv_rows = {}
    for t_start in t_starts:
        for tmpl, dm, diff in variants:
            key = tmpl.replace("{t}", str(t_start))
            results[key]  = []
            csv_rows[key] = [f"sample_idx,{key}"]

    for c, idx in enumerate(idxs, start=1):
        true      = test_fields[idx]
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps,
                                     seed=args.seed + idx)
        x_obs = true.copy(); x_obs[:, ~path_mask] = 0.0

        for t_start in t_starts:
            for tmpl, dm, diff in variants:
                key = tmpl.replace("{t}", str(t_start))
                pred = ae_warmstart_repaint(ae_m, dm, diff,
                                            x_obs, path_mask, land_mask,
                                            device=device, t_start=t_start)
                rmse = rmse_ocean(pred, true, ocean_mask)
                results[key].append(rmse)
                csv_rows[key].append(f"{idx},{rmse:.8f}")

        if c % 5 == 0 or c == n_samples:
            print(f"  [{c}/{n_samples}]", end="")
            for t_start in t_starts:
                for tmpl, _, _ in variants:
                    key = tmpl.replace("{t}", str(t_start))
                    print(f"  {key.split('_', 2)[2]}={results[key][-1]:.4f}", end="")
            print()

    # ── Summary ──────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    lines = [
        "AE Warm-Start RePaint Evaluation",
        f"n_samples={n_samples}  path_steps={args.path_steps}  seed={args.seed}",
        f"t_starts={t_starts}",
        "",
        f"{'Method':<40} {'Mean RMSE':>12} {'Std':>10} {'Min':>10} {'Max':>10}",
        "-" * 80,
    ]
    csv_summary = ["model,mean_rmse,std_rmse,min_rmse,max_rmse,n_samples"]

    for key, vals in results.items():
        arr = np.array(vals)
        m, s, mn, mx = arr.mean(), arr.std(), arr.min(), arr.max()
        lines.append(f"{key:<40} {m:12.6f} {s:10.6f} {mn:10.6f} {mx:10.6f}")
        csv_summary.append(f"{key},{m:.8f},{s:.8f},{mn:.8f},{mx:.8f},{n_samples}")
        with open(os.path.join(args.out_dir, f"{key}_per_sample.csv"), "w") as f:
            f.write("\n".join(csv_rows[key]) + "\n")

    summary_str = "\n".join(lines)
    print("\n" + summary_str)

    with open(os.path.join(args.out_dir, "ae_warmstart_summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    with open(os.path.join(args.out_dir, "ae_warmstart_summary.csv"), "w") as f:
        f.write("\n".join(csv_summary) + "\n")

    print(f"\nSaved to {args.out_dir}/ae_warmstart_summary.txt")


if __name__ == "__main__":
    main()
