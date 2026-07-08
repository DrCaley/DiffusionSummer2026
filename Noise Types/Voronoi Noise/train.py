"""
Train the Voronoi-correlated noise DDPM model.

Noise schedule : linear (beta_1=1e-4 → beta_T=0.02)
Noise type     : Voronoi-correlated — spatially piecewise-constant Gaussian.
                 N_SEEDS random seed points partition the grid; all cells in
                 a Voronoi region share one noise draw.

Early stopping : stops when val loss does not improve by --min_delta over
                 --patience consecutive epochs.

Saved checkpoints:
    checkpoints/best_model.pt        — best validation loss
    checkpoints/ckpt_epochNNNN.pt    — saved every --save_every epochs

Usage (run from workspace root):
    python "Voronoi Noise/train.py" --pickle /root/model_pink_noise/data.pickle
    python "Voronoi Noise/train.py" --pickle ... --n_seeds 50 --patience 50
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "utils"))
sys.path.insert(0, _HERE)

import torch
from torch.utils.data import DataLoader

from dataset       import OceanCurrentDataset
from diffusion     import DDPM, CURL_DIV_WEIGHT, NOISE_TYPE
from repaint_model import Repaint

SCHEDULE = "linear"


def parse_args():
    p = argparse.ArgumentParser(description="Train Voronoi-correlated noise DDPM.")
    p.add_argument("--pickle",     required=True)
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--time_dim",   type=int,   default=256)
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--n_seeds",    type=int,   default=50,
                   help="Number of Voronoi seed points per noise sample. "
                        "Controls spatial correlation scale (~H*W/n_seeds cells each).")
    p.add_argument("--save_dir",   default=None)
    p.add_argument("--resume",     default=None)
    p.add_argument("--patience",   type=int,   default=50)
    p.add_argument("--min_delta",  type=float, default=1e-5)
    p.add_argument("--save_every", type=int,   default=50)
    p.add_argument("--workers",    type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_dir is None:
        args.save_dir = os.path.join(_HERE, "checkpoints")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device          : {device}")
    print(f"Schedule        : {SCHEDULE}")
    print(f"Noise type      : {NOISE_TYPE}")
    print(f"N_seeds         : {args.n_seeds}")
    print(f"Loss            : eps_mse + {CURL_DIV_WEIGHT} * curl_div")
    print(f"Save dir        : {args.save_dir}")

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    land_mask    = train_ds.land_mask.to(device)
    land_mask_np = train_ds.land_mask.numpy()

    ocean_pixels = train_ds.data[:, :, ~train_ds.land_mask]
    noise_std    = float(ocean_pixels.std())
    print(f"noise_std       : {noise_std:.5f}  (ocean pixel std)")

    H, W = train_ds.data.shape[2], train_ds.data.shape[3]
    n_ocean = int((~train_ds.land_mask).sum())
    avg_cell_size = n_ocean / args.n_seeds
    print(f"Grid            : {H}×{W}, ocean cells={n_ocean}")
    print(f"Avg cell size   : {avg_cell_size:.1f} grid cells per Voronoi region")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model + diffusion ----
    model = Repaint(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters      : {n_params:,}")

    diffusion = DDPM(T=args.T, device=device, noise_std=noise_std,
                     land_mask_np=land_mask_np, n_seeds=args.n_seeds)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Resume ----
    start_epoch = 0
    best_val    = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "val_loss" in ckpt:
            best_val = ckpt["val_loss"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.5f}")

    # ---- Early stopping state ----
    patience_counter = 0

    # ---- Training loop ----
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x0 in train_loader:
            x0 = x0.to(device)
            loss = diffusion.training_loss(model, x0, land_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                val_loss += diffusion.training_loss(model, x0, land_mask).item()
        val_loss /= len(val_loader)

        scheduler.step()
        improved = val_loss < (best_val - args.min_delta)
        print(
            f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | "
            f"val={val_loss:.5f} | patience={patience_counter}/{args.patience}"
        )

        ckpt_data = {
            "epoch":           epoch,
            "model":           model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "val_loss":        val_loss,
            "noise_std":       noise_std,
            "noise_type":      NOISE_TYPE,
            "n_seeds":         args.n_seeds,
            "schedule":        SCHEDULE,
            "curl_div_weight": CURL_DIV_WEIGHT,
            "args":            vars(args),
        }

        if improved:
            best_val = val_loss
            patience_counter = 0
            torch.save(ckpt_data, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  --> new best saved (val={best_val:.5f})")
        else:
            patience_counter += 1

        if epoch % args.save_every == 0:
            torch.save(
                ckpt_data,
                os.path.join(args.save_dir, f"ckpt_epoch{epoch:04d}.pt"),
            )

        if patience_counter >= args.patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch} "
                f"(no improvement for {args.patience} epochs)."
            )
            break

    print(f"\nTraining complete.  Best val loss: {best_val:.5f}")
    print(f"Best checkpoint : {args.save_dir}/best_model.pt")


if __name__ == "__main__":
    main()
