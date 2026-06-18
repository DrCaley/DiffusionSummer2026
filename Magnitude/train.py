"""
Train the UNet speed (magnitude) regressor.

Learns to reconstruct a dense ocean-current speed field from sparse robot-path
observations.  Each epoch every snapshot is revealed through a freshly sampled
random walk, so the network learns a general sparse→dense prior rather than
memorizing a fixed observation pattern.

Loss: masked MSE over ocean cells (land excluded), in standardized speed space.

Usage:
    python Magnitude/train.py --pickle Datasets/data.pickle \
        --epochs 200 --batch 16 --path_steps 150 --save_dir Magnitude/checkpoints
"""

import argparse
import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from mag_dataset import MagnitudeDataset, speed_stats
from model       import MagnitudeUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="Datasets/data.pickle")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--path_steps", type=int,   default=150)
    p.add_argument("--unobs_weight", type=float, default=1.0,
                   help="extra loss weight on UNobserved ocean cells (1.0 = uniform)")
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="linear LR warmup epochs before cosine decay")
    p.add_argument("--min_lr_frac", type=float, default=0.02,
                   help="cosine decays LR to this fraction of --lr")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay for weight averaging (<=0 disables EMA)")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="max grad norm (<=0 disables clipping)")
    p.add_argument("--save_dir",   default="Magnitude/checkpoints")
    p.add_argument("--device",     default=None, help="cuda | mps | cpu (auto if unset)")
    return p.parse_args()


class EMA:
    """Exponential moving average of model parameters (CPU-light, in-place swap)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)            # buffers / int tensors: just track

    def store_and_copy_to(self, model: torch.nn.Module):
        """Back up live weights and load EMA weights for eval/save."""
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: torch.nn.Module):
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None


def lr_factor(epoch: int, warmup: int, total: int, min_frac: float) -> float:
    """Linear warmup then cosine decay to min_frac (epoch is 1-indexed)."""
    if warmup > 0 and epoch <= warmup:
        return epoch / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_frac + (1.0 - min_frac) * cos


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_mse(pred, target, ocean, path_ch, unobs_weight):
    """MSE over ocean cells, optionally up-weighting unobserved cells."""
    err2 = (pred - target) ** 2 * ocean
    if unobs_weight != 1.0:
        # path_ch is 1 at observed cells; unobserved ocean = ocean * (1 - path)
        w = ocean * (1.0 + (unobs_weight - 1.0) * (1.0 - path_ch))
        err2 = (pred - target) ** 2 * w
        return err2.sum() / (w.sum() + 1e-8)
    return err2.sum() / (ocean.sum() + 1e-8)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = pick_device(args.device)
    print(f"Device: {device}")

    # ---- Speed statistics from the training split ----
    print("Computing training-split speed statistics...")
    speed_mean, speed_std = speed_stats(args.pickle, split=0)
    print(f"  speed mean = {speed_mean:.4f}   std = {speed_std:.4f}")

    train_ds = MagnitudeDataset(args.pickle, split=0, speed_mean=speed_mean,
                                speed_std=speed_std, path_steps=args.path_steps,
                                fixed_paths=False)
    val_ds   = MagnitudeDataset(args.pickle, split=1, speed_mean=speed_mean,
                                speed_std=speed_std, path_steps=args.path_steps,
                                fixed_paths=True, seed=1234)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers)

    model = MagnitudeUNet(in_ch=3, base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_ema = args.ema_decay > 0.0
    ema = EMA(model, args.ema_decay) if use_ema else None
    print(f"LR: warmup {args.warmup_epochs} ep -> cosine to {args.min_lr_frac:g}x | "
          f"EMA: {'on (' + str(args.ema_decay) + ')' if use_ema else 'off'} | "
          f"grad_clip: {args.grad_clip if args.grad_clip > 0 else 'off'}")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- LR schedule (per-epoch) ----
        cur_lr = args.lr * lr_factor(epoch, args.warmup_epochs, args.epochs, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        # ---- Train ----
        model.train()
        tr_loss, tr_n = 0.0, 0
        for inp, target, ocean in train_dl:
            inp, target, ocean = inp.to(device), target.to(device), ocean.to(device)
            path_ch = inp[:, 1:2]
            pred = model(inp)
            loss = masked_mse(pred, target, ocean, path_ch, args.unobs_weight)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if use_ema:
                ema.update(model)
            tr_loss += loss.item() * inp.size(0)
            tr_n    += inp.size(0)
        tr_loss /= max(tr_n, 1)

        # ---- Validate (on EMA weights when enabled) ----
        if use_ema:
            ema.store_and_copy_to(model)
        model.eval()
        va_loss, va_n = 0.0, 0
        va_rmse_phys = 0.0
        with torch.no_grad():
            for inp, target, ocean in val_dl:
                inp, target, ocean = inp.to(device), target.to(device), ocean.to(device)
                path_ch = inp[:, 1:2]
                pred = model(inp)
                loss = masked_mse(pred, target, ocean, path_ch, args.unobs_weight)
                va_loss += loss.item() * inp.size(0)
                va_n    += inp.size(0)
                # Physical-units RMSE (un-standardize: multiply by std)
                err2 = ((pred - target) ** 2 * ocean).sum().item()
                cnt  = ocean.sum().item()
                va_rmse_phys += err2 / max(cnt, 1) * inp.size(0)
        va_loss /= max(va_n, 1)
        va_rmse_phys = np.sqrt(va_rmse_phys / max(va_n, 1)) * speed_std

        # model.state_dict() here holds EMA weights (swapped in above) when enabled,
        # so checkpoints persist the exact weights that produced these val metrics.
        flag = ""
        if va_loss < best_val:
            best_val = va_loss
            flag = " *"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_loss": va_loss,
                "speed_mean": speed_mean,
                "speed_std": speed_std,
                "args": vars(args),
            }, os.path.join(args.save_dir, "best_magnitude_unet.pt"))

        if epoch % 25 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_loss": va_loss,
                "speed_mean": speed_mean,
                "speed_std": speed_std,
                "args": vars(args),
            }, os.path.join(args.save_dir, f"ckpt_ep{epoch:04d}_magnitude_unet.pt"))

        # Restore live (non-EMA) weights so training continues normally.
        if use_ema:
            ema.restore(model)

        print(f"Epoch {epoch:3d}/{args.epochs} | lr={cur_lr:.2e} | train={tr_loss:.5f} | "
              f"val={va_loss:.5f} | val_RMSE(phys)={va_rmse_phys:.4f}{flag}",
              flush=True)

    print(f"\nDone. Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
