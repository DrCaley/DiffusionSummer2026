"""
Train the CONDITIONED magnitude (speed) UNet — Phase 2.

Same architecture and training machinery as ``Magnitude/train.py`` (masked MSE,
warmup+cosine LR, EMA, grad clip), but the regressor now ingests the full
10-channel conditioning stack used by the diffusion model (obs + temporal priors
+ geometry) instead of the 3-channel [obs_speed, path_mask, land] input.  The
temporal priors give the network a frame-energy signal, fixing the far-field
speed overshoot of the original 3-channel model.

The output convention is UNCHANGED (standardized by physical speed_mean/speed_std)
so the existing two-head fusion code works without modification.

Usage (server):
    python3 Magnitude/train_cond_mag.py \
        --pickle Datasets/data_divfree_chrono.pickle \
        --lags 13,25 --path_steps 30,200 \
        --epochs 200 --batch 16 --save_dir Magnitude/checkpoints_cond_mag
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from cond_mag_dataset import CondMagnitudeDataset, speed_stats_chrono
from cond_dataset     import cond_channels, ConditionalOceanDataset
from model            import MagnitudeUNet
# Reuse the already-tested training helpers (no duplication).
from train            import EMA, lr_factor, pick_device, masked_mse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle", default="Datasets/data_divfree_chrono.pickle")
    p.add_argument("--lags", default="13,25", help="comma-sep prior lags")
    p.add_argument("--path_steps", default="30,200",
                   help="int for fixed path length, or 'min,max' to sample per item")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--unobs_weight", type=float, default=1.0,
                   help="extra loss weight on UNobserved ocean cells (1.0 = uniform)")
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_frac", type=float, default=0.02)
    p.add_argument("--ema_decay",  type=float, default=0.999)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--save_dir",   default="Magnitude/checkpoints_cond_mag")
    p.add_argument("--device",     default=None, help="cuda | mps | cpu (auto if unset)")
    return p.parse_args()


def parse_path_steps(s: str):
    parts = [int(x) for x in str(s).split(",") if x.strip()]
    return tuple(parts) if len(parts) == 2 else parts[0]


def save_ckpt(path, epoch, model, va_loss, speed_mean, speed_std,
              data_std, lags, cond_ch, args):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "val_loss": va_loss,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "data_std": data_std,
        "lags": lags,
        "cond_ch": cond_ch,
        "args": vars(args),
    }, path)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = pick_device(args.device)
    print(f"Device: {device}")

    lags = tuple(int(x) for x in args.lags.split(","))
    path_steps = parse_path_steps(args.path_steps)
    cond_ch = cond_channels(lags)

    # data_std (vector-component std) comes from the chrono pickle; speed stats
    # are computed in physical units over the train split.
    data_mean, data_std = ConditionalOceanDataset.compute_stats(args.pickle, split=0)
    print(f"data_std = {data_std:.5f}  (component normalization)")
    print("Computing training-split physical speed statistics...")
    speed_mean, speed_std = speed_stats_chrono(args.pickle, data_std, lags, split=0)
    print(f"  speed mean = {speed_mean:.4f}   std = {speed_std:.4f}")
    print(f"cond channels = {cond_ch}   lags = {lags}   path_steps = {path_steps}")

    train_ds = CondMagnitudeDataset(args.pickle, split=0, speed_mean=speed_mean,
                                    speed_std=speed_std, data_std=data_std,
                                    lags=lags, path_steps=path_steps,
                                    deterministic=False)
    val_ds   = CondMagnitudeDataset(args.pickle, split=1, speed_mean=speed_mean,
                                    speed_std=speed_std, data_std=data_std,
                                    lags=lags, path_steps=path_steps,
                                    deterministic=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers)

    model = MagnitudeUNet(in_ch=cond_ch, base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_ema = args.ema_decay > 0.0
    ema = EMA(model, args.ema_decay) if use_ema else None
    print(f"LR: warmup {args.warmup_epochs} ep -> cosine to {args.min_lr_frac:g}x | "
          f"EMA: {'on (' + str(args.ema_decay) + ')' if use_ema else 'off'} | "
          f"grad_clip: {args.grad_clip if args.grad_clip > 0 else 'off'}")

    best_path = os.path.join(args.save_dir, "best_cond_magnitude_unet.pt")
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        cur_lr = args.lr * lr_factor(epoch, args.warmup_epochs, args.epochs, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        model.train()
        tr_loss, tr_n = 0.0, 0
        for cond, target, ocean in train_dl:
            cond, target, ocean = cond.to(device), target.to(device), ocean.to(device)
            path_ch = cond[:, 2:3]                  # path_mask is channel 2 of cond
            pred = model(cond)
            loss = masked_mse(pred, target, ocean, path_ch, args.unobs_weight)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if use_ema:
                ema.update(model)
            tr_loss += loss.item() * cond.size(0)
            tr_n    += cond.size(0)
        tr_loss /= max(tr_n, 1)

        if use_ema:
            ema.store_and_copy_to(model)
        model.eval()
        va_loss, va_n, va_rmse_phys = 0.0, 0, 0.0
        with torch.no_grad():
            for cond, target, ocean in val_dl:
                cond, target, ocean = cond.to(device), target.to(device), ocean.to(device)
                path_ch = cond[:, 2:3]
                pred = model(cond)
                loss = masked_mse(pred, target, ocean, path_ch, args.unobs_weight)
                va_loss += loss.item() * cond.size(0)
                va_n    += cond.size(0)
                err2 = ((pred - target) ** 2 * ocean).sum().item()
                cnt  = ocean.sum().item()
                va_rmse_phys += err2 / max(cnt, 1) * cond.size(0)
        va_loss /= max(va_n, 1)
        va_rmse_phys = np.sqrt(va_rmse_phys / max(va_n, 1)) * speed_std

        flag = ""
        if va_loss < best_val:
            best_val = va_loss
            flag = " *"
            save_ckpt(best_path, epoch, model, va_loss, speed_mean, speed_std,
                      data_std, lags, cond_ch, args)
        if epoch % 25 == 0:
            save_ckpt(os.path.join(args.save_dir, f"ckpt_ep{epoch:04d}_cond_magnitude_unet.pt"),
                      epoch, model, va_loss, speed_mean, speed_std,
                      data_std, lags, cond_ch, args)

        if use_ema:
            ema.restore(model)

        print(f"Epoch {epoch:3d}/{args.epochs} | lr={cur_lr:.2e} | train={tr_loss:.5f} | "
              f"val={va_loss:.5f} | val_RMSE(phys)={va_rmse_phys:.4f}{flag}", flush=True)

    print(f"\nDone. Best val loss: {best_val:.5f}  ->  {best_path}")


if __name__ == "__main__":
    main()
