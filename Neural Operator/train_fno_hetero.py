"""
train_fno_hetero.py
=====================
Trains the heteroscedastic single-shot FNO (model_fno_hetero.FNOHetero) as a
direct masked-reconstruction regressor: given sparse path observations,
predict a per-pixel (mean, variance) for the full field in one forward pass
— no diffusion timesteps, no iterative sampling.

A fresh biased_walk_path mask is drawn for every sample in every batch (CPU,
numpy), matching the same path-generation convention (n_steps=150) used
everywhere else in this project, so the model sees a wide distribution of
path shapes/lengths/positions during training.

Loss: Gaussian negative log-likelihood, masked to ocean pixels only:
    NLL = 0.5 * (log_var + (x0_true - mean)^2 / exp(log_var))

Usage:
    python3 train_fno_hetero.py --pickle /root/ocean_ddpm/data_local.pickle \\
        --epochs 150
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)


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


def parse_args():
    p = argparse.ArgumentParser(description="Train heteroscedastic single-shot FNO.")
    p.add_argument("--pickle",        default="/root/ocean_ddpm/data_local.pickle")
    p.add_argument("--diffusion_dir", default=None)
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--patience",      type=int,   default=20,
                   help="Stop after this many epochs with no val_nll improvement "
                        "(early stopping — matches the convention in train_fno.py).")
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--width",         type=int,   default=64)
    p.add_argument("--modes1",        type=int,   default=16)
    p.add_argument("--modes2",        type=int,   default=16)
    p.add_argument("--n_layers",      type=int,   default=4)
    p.add_argument("--path_steps",    type=int,   default=150)
    p.add_argument("--mask_pool_size", type=int,  default=4096)
    p.add_argument("--save_dir",      default=None)
    p.add_argument("--workers",       type=int,   default=0)
    p.add_argument("--ckpt_every",    type=int,   default=10)
    return p.parse_args()


def main():
    args = parse_args()

    diff_dir = _find_diffusion_dir(args.diffusion_dir)
    sys.path.insert(0, diff_dir)
    print(f"Using dataset helpers from: {diff_dir}")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from dataset        import OceanCurrentDataset
    from repaint_infer  import biased_walk_path
    from model_fno_hetero import FNOHetero, build_input

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_dir is None:
        args.save_dir = os.path.join(_SCRIPT_DIR, "checkpoints_fno_hetero")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Save dir   : {args.save_dir}")

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    land_mask_np = train_ds.land_mask.numpy()               # (H, W) bool
    ocean_t = torch.from_numpy(~land_mask_np).float()[None, None].to(device)  # (1,1,H,W)

    # Precompute a large pool of random-walk masks ONCE — generating a fresh
    # mask per sample per batch via the non-vectorized Python loop inside
    # biased_walk_path is the dominant cost otherwise (~9600 calls/epoch at
    # batch=32, ~55s/epoch of pure CPU mask generation). Sampling from a
    # precomputed pool turns that into a single vectorized gather.
    print(f"Precomputing {args.mask_pool_size} random-walk masks (n_steps={args.path_steps})...")
    mask_pool = np.stack([
        biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=None)
        for _ in range(args.mask_pool_size)
    ])
    mask_pool_t = torch.from_numpy(mask_pool).float().to(device)  # (N_pool, H, W)
    print(f"Mask pool ready: {tuple(mask_pool_t.shape)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model ----
    model = FNOHetero(
        in_ch=4, out_ch=2, width=args.width, modes1=args.modes1,
        modes2=args.modes2, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    def make_batch_input(x0_batch):
        B, C, H, W = x0_batch.shape
        idx = torch.randint(0, mask_pool_t.shape[0], (B,), device=device)
        path_t = mask_pool_t[idx].unsqueeze(1)  # (B,1,H,W)
        x0_obs = x0_batch * path_t
        ocean_b = ocean_t.expand(B, -1, -1, -1)
        return build_input(x0_obs, path_t, ocean_b)

    def gaussian_nll(mean, log_var, target):
        var = log_var.exp()
        nll = 0.5 * (log_var + (target - mean) ** 2 / var)
        return (nll * ocean_t).sum() / (ocean_t.sum() * target.shape[1] * target.shape[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Reacts to validation loss directly (not tied to a fixed epoch count) —
    # the right fit when training runs until it plateaus rather than for a
    # fixed budget.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6,
    )

    best_val  = float("inf")
    best_name = "best_fno_hetero.pt"
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x0 in train_loader:
            x0 = x0.to(device)
            inp = make_batch_input(x0)
            mean, log_var = model(inp)
            loss = gaussian_nll(mean, log_var, x0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                inp = make_batch_input(x0)
                mean, log_var = model(inp)
                val_loss += gaussian_nll(mean, log_var, x0).item()
                ocean_b = ocean_t.expand_as(mean)
                val_rmse += float(torch.sqrt(
                    ((mean - x0) ** 2 * ocean_b).sum() / (ocean_b.sum())
                ))
        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:4d}/{args.epochs} | train_nll={train_loss:.5f} | "
              f"val_nll={val_loss:.5f} | val_mean_rmse={val_rmse:.5f} | lr={lr_now:.2e}",
              flush=True)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_loss": val_loss, "val_mean_rmse": val_rmse, "args": vars(args)},
                os.path.join(args.save_dir, best_name),
            )
            print(f"  -> saved new best (val_nll={val_loss:.5f}, val_mean_rmse={val_rmse:.5f})",
                  flush=True)
        else:
            epochs_no_improve += 1
            print(f"  no improvement for {epochs_no_improve}/{args.patience} epochs", flush=True)

        if epoch % args.ckpt_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                os.path.join(args.save_dir, f"ckpt_fno_hetero_epoch{epoch:04d}.pt"),
            )

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs "
                  f"(loss has plateaued / is oscillating). Best val NLL: {best_val:.5f}")
            break

    else:
        print(f"\nReached --epochs cap ({args.epochs}) without early stopping.")

    print(f"Training complete. Best val NLL: {best_val:.5f}")
    print(f"Best checkpoint: {args.save_dir}/{best_name}")


if __name__ == "__main__":
    main()
