"""
Train a HETEROSCEDASTIC variance head on top of the conditioned magnitude UNet.

Motivation
----------
The deterministic magnitude UNet gives one speed map per frame, so every fused
diffusion draw is forced to the SAME magnitude.  That zeroes the magnitude
spread, capping r_magnitude / r_overall far below r_angle (which is high only
because DIRECTION is genuinely stochastic).  You cannot calibrate the
uncertainty of a quantity that does not vary.

Fix: predict per-cell speed as a Gaussian  speed ~ N(mu(x), sigma(x)^2)  and
sample each draw's magnitude from it.  To keep ACCURACY identical to the current
model, we warm-start the whole backbone + mean head from the trained
``Cond_Magnitude_UNet.pt`` and FREEZE it; only the new 1×1 log-variance head is
trained, with a Gaussian negative-log-likelihood loss.  The mean (hence RMSE) is
unchanged by construction; we only learn a calibrated spread.

Usage:
    python3 Magnitude/train_cond_mag_hetero.py \
        --init_checkpoint Models/Cond_Magnitude_UNet.pt \
        --pickle Datasets/pickles/data_divfree_chrono.pickle \
        --lags 13,25 --path_steps 30,200 \
        --epochs 30 --batch 16 \
        --save_dir Magnitude/checkpoints_cond_mag_hetero
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
from model            import HeteroMagnitudeUNet
from train            import EMA, lr_factor, pick_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_checkpoint", default="Models/Cond_Magnitude_UNet.pt",
                   help="trained deterministic magnitude UNet to warm-start mean+backbone")
    p.add_argument("--pickle", default="Datasets/pickles/data_divfree_chrono.pickle")
    p.add_argument("--lags", default="13,25", help="comma-sep prior lags")
    p.add_argument("--path_steps", default="30,200",
                   help="int for fixed path length, or 'min,max' to sample per item")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=2e-3,
                   help="LR for the variance head (the only trained params)")
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--head_hidden", type=int,  default=32,
                   help="hidden channels in the variance head (0 = original 1x1 "
                        "conv; >0 = two 3x3 convs for spatially-coherent sigma)")
    p.add_argument("--smooth_weight", type=float, default=0.05,
                   help="weight of the total-variation smoothness prior on logvar "
                        "(pushes the sigma field toward the smooth empirical "
                        "posterior the calibration metric scores against)")
    p.add_argument("--train_backbone", action="store_true",
                   help="also fine-tune the backbone+mean (default: frozen, "
                        "guaranteeing identical accuracy)")
    p.add_argument("--logvar_init", type=float, default=-2.0)
    p.add_argument("--logvar_min", type=float, default=-8.0)
    p.add_argument("--logvar_max", type=float, default=4.0)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--min_lr_frac", type=float, default=0.05)
    p.add_argument("--ema_decay",  type=float, default=0.0)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--save_dir",   default="Magnitude/checkpoints_cond_mag_hetero")
    p.add_argument("--device",     default=None, help="cuda | mps | cpu (auto if unset)")
    return p.parse_args()


def parse_path_steps(s: str):
    parts = [int(x) for x in str(s).split(",") if x.strip()]
    return tuple(parts) if len(parts) == 2 else parts[0]


def gaussian_nll(mean, logvar, target, ocean, lv_min, lv_max):
    """
    Masked Gaussian negative log-likelihood over ocean cells.
        nll = 0.5 * ( logvar + (target-mean)^2 * exp(-logvar) )
    The constant 0.5*log(2*pi) is dropped (irrelevant to optimization).
    """
    logvar = logvar.clamp(lv_min, lv_max)
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (target - mean) ** 2 * inv_var)
    nll = nll * ocean
    return nll.sum() / (ocean.sum() + 1e-8)


def tv_smoothness(logvar, ocean):
    """
    Masked total-variation penalty on the log-variance field: mean absolute
    spatial gradient over ocean cell pairs.  Encourages a smooth, coherent sigma
    map (matching the smooth empirical neighbour-posterior spread) instead of the
    salt-and-pepper field the pointwise NLL alone produces.
    """
    dx = (logvar[:, :, :, 1:] - logvar[:, :, :, :-1]).abs()
    dy = (logvar[:, :, 1:, :] - logvar[:, :, :-1, :]).abs()
    mx = ocean[:, :, :, 1:] * ocean[:, :, :, :-1]
    my = ocean[:, :, 1:, :] * ocean[:, :, :-1, :]
    tx = (dx * mx).sum() / (mx.sum() + 1e-8)
    ty = (dy * my).sum() / (my.sum() + 1e-8)
    return tx + ty


def save_ckpt(path, epoch, model, va_nll, speed_mean, speed_std,
              data_std, lags, cond_ch, args):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "val_nll": va_nll,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "data_std": data_std,
        "lags": lags,
        "cond_ch": cond_ch,
        "hetero": True,
        "logvar_clip": [args.logvar_min, args.logvar_max],
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

    model = HeteroMagnitudeUNet(in_ch=cond_ch, base_ch=args.base_ch,
                                logvar_init=args.logvar_init,
                                head_hidden=args.head_hidden).to(device)

    # ---- warm-start mean + backbone from the deterministic UNet ----
    init = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(init["model"], strict=False)
    loaded = [k for k in init["model"] if k in dict(model.named_parameters())
              or k in dict(model.named_buffers())]
    print(f"warm-start from {os.path.basename(args.init_checkpoint)}: "
          f"loaded {len(loaded)} tensors; new (variance head) = {missing}")

    # ---- freeze everything except the log-variance head ----
    if not args.train_backbone:
        trained = []
        for name, p in model.named_parameters():
            if name.startswith("logvar_conv") or name.startswith("logvar_head"):
                p.requires_grad_(True); trained.append(name)
            else:
                p.requires_grad_(False)
        print(f"FROZEN backbone+mean; training only: {trained}")
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"Trainable parameters: {n_train:,}")

    opt = torch.optim.AdamW(params, lr=args.lr)
    use_ema = args.ema_decay > 0.0
    ema = EMA(model, args.ema_decay) if use_ema else None

    best_path = os.path.join(args.save_dir, "best_cond_magnitude_hetero.pt")
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        cur_lr = args.lr * lr_factor(epoch, args.warmup_epochs, args.epochs, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        model.train()
        tr_loss, tr_n = 0.0, 0
        for cond, target, ocean in train_dl:
            cond, target, ocean = cond.to(device), target.to(device), ocean.to(device)
            mean, logvar = model(cond)
            loss = gaussian_nll(mean, logvar, target, ocean,
                                args.logvar_min, args.logvar_max)
            if args.smooth_weight > 0:
                loss = loss + args.smooth_weight * tv_smoothness(logvar, ocean)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if use_ema:
                ema.update(model)
            tr_loss += loss.item() * cond.size(0)
            tr_n    += cond.size(0)
        tr_loss /= max(tr_n, 1)

        if use_ema:
            ema.store_and_copy_to(model)
        model.eval()
        va_loss, va_n, va_sig_phys = 0.0, 0, 0.0
        with torch.no_grad():
            for cond, target, ocean in val_dl:
                cond, target, ocean = cond.to(device), target.to(device), ocean.to(device)
                mean, logvar = model(cond)
                loss = gaussian_nll(mean, logvar, target, ocean,
                                    args.logvar_min, args.logvar_max)
                va_loss += loss.item() * cond.size(0)
                va_n    += cond.size(0)
                sig = torch.exp(0.5 * logvar.clamp(args.logvar_min, args.logvar_max))
                va_sig_phys += (sig * ocean).sum().item() / max(ocean.sum().item(), 1) * cond.size(0)
        va_loss /= max(va_n, 1)
        va_sig_phys = va_sig_phys / max(va_n, 1) * speed_std

        flag = ""
        if va_loss < best_val:
            best_val = va_loss
            flag = " *"
            save_ckpt(best_path, epoch, model, va_loss, speed_mean, speed_std,
                      data_std, lags, cond_ch, args)

        if use_ema:
            ema.restore(model)

        print(f"Epoch {epoch:3d}/{args.epochs} | lr={cur_lr:.2e} | "
              f"train_nll={tr_loss:.5f} | val_nll={va_loss:.5f} | "
              f"mean_sigma(phys)={va_sig_phys:.4f}{flag}", flush=True)

    print(f"\nDone. Best val NLL: {best_val:.5f}  ->  {best_path}")


if __name__ == "__main__":
    main()
