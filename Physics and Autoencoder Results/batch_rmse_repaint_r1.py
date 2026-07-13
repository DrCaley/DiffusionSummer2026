import argparse
import os
import pickle
import numpy as np
import torch

from diffusion import DDPM
from repaint_model import Repaint
from ae_model import RepaintAutoencoder
from atmodist_model import AtmoDistEncoder


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
            if dot == 1:
                w = straight_bias
            elif dot == 0:
                w = side
            else:
                w = side * 0.05
            nr, nc = r + dr, c + dc
            novelty = 1.0 / (1.0 + visit_count[nr, nc])
            weights.append(w * novelty)

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        idx = rng.choice(len(valid), p=weights)
        dr, dc = valid[idx]
        r, c = r + dr, c + dc
        cur_dir = (dr, dc)
        visit_count[r, c] += 1.0
        path_mask[r, c] = True

    return path_mask


@torch.no_grad()
def repaint_infer_r1(model, diffusion, x0_known, path_mask, land_mask, device="cpu"):
    """RePaint inference with resample=1, stride=1."""
    model.eval()
    H, W = x0_known.shape[1:]

    x0_known = x0_known.unsqueeze(0).to(device)
    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std
    xt = xt * ocean_t

    T = diffusion.T
    for t_int in reversed(range(T)):
        t_prev_int = max(t_int - 1, 0)
        xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)

        t_prev_tensor = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
        xt_known, _ = diffusion.q_sample(x0_known, t_prev_tensor)

        xt = known_t * xt_known + (1.0 - known_t) * xt_unknown
        xt = xt * ocean_t

    return xt.squeeze(0).cpu().numpy()  # (2,H,W)


def rmse_ocean(pred, true, ocean_mask):
    return float(np.sqrt(np.mean((pred[:, ocean_mask] - true[:, ocean_mask]) ** 2)))


def build_input_autoencoder(x0_true, path_mask):
    x_obs = x0_true.copy()
    x_obs[:, ~path_mask] = 0.0
    mask_ch = path_mask.astype(np.float32)[None, :, :]
    return np.concatenate([x_obs, mask_ch], axis=0)


def load_timecond_split(data, split_name):
    # Handle both string-keyed ("train"/"val"/"test") and integer-keyed (0/1/2) pickles
    SPLIT_IDX = {"train": 0, "val": 1, "test": 2}
    if isinstance(data, dict) and split_name in data:
        key = split_name
    else:
        key = SPLIT_IDX[split_name]
    arr = np.asarray(data[key], dtype=np.float32)  # (H,W,2,N)
    fields = np.transpose(arr, (3, 2, 0, 1)).astype(np.float32)  # (N,2,H,W)
    land_mask = np.isnan(arr[:, :, 0, 0])
    fields = np.nan_to_num(fields, nan=0.0)
    return fields, land_mask


def evaluate_models(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    train_fields, train_land = load_timecond_split(data, "train")
    test_fields, test_land = load_timecond_split(data, "test")

    if not np.array_equal(train_land, test_land):
        print("Warning: train/test land masks differ; using test land mask for RMSE")

    land_mask = test_land
    ocean_mask = ~land_mask

    n_total = test_fields.shape[0]
    n_samples = min(args.n_samples, n_total - args.sample_start)
    idxs = list(range(args.sample_start, args.sample_start + n_samples))

    # ----- Load base/physics models (RePaint UNet + diffusion) -----
    def load_repaint_and_diffusion(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        base_ch = ckpt_args.get("base_ch", 64)
        time_dim = ckpt_args.get("time_dim", 256)
        T = ckpt_args.get("T", 1000)
        schedule = ckpt.get("schedule", "linear")
        noise_std = ckpt.get("noise_std", None)

        if noise_std is None:
            noise_std = float(train_fields[:, :, ocean_mask].std())

        model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        diffusion = DDPM(T=T, beta_schedule=schedule, device=device, noise_std=noise_std)
        return model, diffusion

    base_model, base_diff = load_repaint_and_diffusion(args.base_ckpt)
    phys_model, phys_diff = load_repaint_and_diffusion(args.physics_ckpt)

    # ----- Load autoencoder -----
    ae_ckpt = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae_args = ae_ckpt.get("args", {})
    ae_base = ae_args.get("base_ch", 64)
    ae_model = RepaintAutoencoder(in_ch=3, out_ch=2, base_ch=ae_base).to(device)
    ae_model.load_state_dict(ae_ckpt["model"])
    ae_model.eval()

    # ----- Load AtmoDist (retrieval-style inference) -----
    atm_ckpt = torch.load(args.atmodist_ckpt, map_location=device, weights_only=False)
    atm_args = atm_ckpt.get("args", {})
    atm_base = atm_args.get("base_ch", 64)
    atm_emb = atm_args.get("emb_dim", 256)
    n_classes = len(atm_ckpt.get("class_ranges", ((1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64))))
    atm_model = AtmoDistEncoder(in_ch=2, base_ch=atm_base, emb_dim=atm_emb, n_classes=n_classes).to(device)
    atm_model.load_state_dict(atm_ckpt["model"])
    atm_model.eval()

    # Precompute train embeddings for retrieval baseline
    train_tensor = torch.from_numpy(train_fields).to(device)
    with torch.no_grad():
        train_emb = []
        bs = 256
        for i in range(0, train_tensor.shape[0], bs):
            train_emb.append(atm_model.encode(train_tensor[i:i + bs]))
        train_emb = torch.cat(train_emb, dim=0)  # (N_train, emb_dim)
        train_emb = torch.nn.functional.normalize(train_emb, dim=1)

    rmses = {
        "base_repaint_r1": [],
        "physics_repaint_r1": [],
        "autoencoder": [],
        "atmodist_retrieval": [],
    }
    per_sample_rows = [
        "sample_idx,base_repaint_r1,physics_repaint_r1,autoencoder,atmodist_retrieval"
    ]

    for c, idx in enumerate(idxs, start=1):
        true = test_fields[idx]  # (2,H,W)
        path_mask = biased_walk_path(land_mask, n_steps=args.path_steps, seed=args.seed + idx)

        # observed sparse field
        x_obs = true.copy()
        x_obs[:, ~path_mask] = 0.0

        # base repaint r=1
        pred_base = repaint_infer_r1(
            base_model, base_diff,
            torch.from_numpy(x_obs),
            path_mask,
            land_mask,
            device=device,
        )
        rmse_base = rmse_ocean(pred_base, true, ocean_mask)
        rmses["base_repaint_r1"].append(rmse_base)

        # physics repaint r=1
        pred_phys = repaint_infer_r1(
            phys_model, phys_diff,
            torch.from_numpy(x_obs),
            path_mask,
            land_mask,
            device=device,
        )
        rmse_phys = rmse_ocean(pred_phys, true, ocean_mask)
        rmses["physics_repaint_r1"].append(rmse_phys)

        # autoencoder
        ae_inp = build_input_autoencoder(true, path_mask)
        with torch.no_grad():
            ae_pred = ae_model(torch.from_numpy(ae_inp).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        rmse_ae = rmse_ocean(ae_pred, true, ocean_mask)
        rmses["autoencoder"].append(rmse_ae)

        # atmodist retrieval: encode sparse observation, nearest train embedding, use neighbor full field
        with torch.no_grad():
            obs_t = torch.from_numpy(x_obs).unsqueeze(0).to(device)
            e = torch.nn.functional.normalize(atm_model.encode(obs_t), dim=1)  # (1,D)
            sim = torch.matmul(train_emb, e.T).squeeze(1)  # (N_train,)
            nn_idx = int(torch.argmax(sim).item())
        pred_atm = train_fields[nn_idx]
        rmse_atm = rmse_ocean(pred_atm, true, ocean_mask)
        rmses["atmodist_retrieval"].append(rmse_atm)

        per_sample_rows.append(
            f"{idx},{rmse_base:.8f},{rmse_phys:.8f},{rmse_ae:.8f},{rmse_atm:.8f}"
        )

        if c % 5 == 0 or c == n_samples:
            print(f"Processed {c}/{n_samples}")

    # aggregate
    lines = []
    lines.append(f"RMSE Evaluation (RePaint resample=1 for diffusion models)")
    lines.append(f"n_samples={n_samples}, sample_start={args.sample_start}, path_steps={args.path_steps}")
    lines.append("")
    lines.append(f"{'Model':<26} {'Mean RMSE':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    lines.append("-" * 78)

    csv_rows = ["model,mean_rmse,std_rmse,min_rmse,max_rmse,n_samples"]

    for k, v in rmses.items():
        arr = np.asarray(v, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        mn = float(arr.min())
        mx = float(arr.max())
        lines.append(f"{k:<26} {mean:12.6f} {std:12.6f} {mn:12.6f} {mx:12.6f}")
        csv_rows.append(f"{k},{mean:.8f},{std:.8f},{mn:.8f},{mx:.8f},{n_samples}")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "rmse_summary.txt")
    csv_path = os.path.join(args.out_dir, "rmse_summary.csv")
    per_sample_path = os.path.join(args.out_dir, "rmse_per_sample.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        f.write("\n".join(per_sample_rows) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {per_sample_path}")


def main():
    p = argparse.ArgumentParser(description="Batch RMSE across base/physics/autoencoder/atmodist")
    p.add_argument("--pickle", default="/root/ocean_ddpm/data.pickle")
    p.add_argument("--base_ckpt", default="/root/autoencoder_train/checkpoints_linear/best_model_linear.pt")
    p.add_argument("--physics_ckpt", default="/root/autoencoder_train/checkpoints_physics/best_model_physics.pt")
    p.add_argument("--ae_ckpt", default="/root/autoencoder_train/checkpoints/best_model_autoencoder.pt")
    p.add_argument("--atmodist_ckpt", default="/root/autoencoder_train/checkpoints_atmodist/best_model_atmodist.pt")
    p.add_argument("--out_dir", default="/root/autoencoder_train/inference_results")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    evaluate_models(args)


if __name__ == "__main__":
    main()
