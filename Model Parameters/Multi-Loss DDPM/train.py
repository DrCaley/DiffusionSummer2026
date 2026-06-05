"""
Training script for the multi-loss DDPM on ocean current fields.

Supports four loss modes via --loss (one or more, space-separated):
  eps          Plain epsilon-MSE only (baseline, same as Basic DDPM)
  curl_div     + curl/divergence penalty (same as Topo DDPM)
  spectral     + FFT power-spectrum penalty
  okubo_weiss  + Okubo-Weiss eddy-structure penalty
  wasserstein  + Sinkhorn–Wasserstein distance between vorticity fields
Usage:
    py train.py --loss spectral --weights 0.0002
    py train.py --loss spectral okubo_weiss --weights 0.0002 0.001
    py train.py --loss curl_div --weights 0.0002
    py train.py --loss wasserstein --weights 1.0 --sinkhorn_blur 0.05
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # project root
sys.path.append(_ROOT)                             # dataset.py
sys.path.append(os.path.join(_ROOT, "DDPM", "model"))  # model.py, diffusion.py

import torch
from torch.utils.data import DataLoader

from dataset   import OceanCurrentDataset
from model     import UNet
from diffusion import DDPM, LOSS_MODES, DEFAULT_WEIGHTS
MultiLossDDPM = DDPM  # alias kept for clarity


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     default="data.pickle")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--time_dim",   type=int,   default=256)
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--noise_type", default="gaussian", choices=["gaussian"],
                   help="Type of noise used in the forward process (default: gaussian)")
    p.add_argument("--schedule",   default="cosine", choices=["cosine", "linear"],
                   help="Noise schedule (default: cosine)")
    p.add_argument("--loss",       default=["spectral"], choices=LOSS_MODES,
                   nargs="+",
                   help="One or more auxiliary loss modes (default: spectral)")
    p.add_argument("--weights",    type=float, default=None, nargs="+",
                   help="Per-loss weights, one per --loss entry in the same order. "
                        "Defaults: spectral=0.0002, curl_div=0.0002, "
                        "okubo_weiss=0.001, wasserstein=1.0")
    p.add_argument("--sinkhorn_blur",  type=float, default=0.05,
                   help="Entropic regularisation blur for Sinkhorn (wasserstein mode only)")
    p.add_argument("--save_dir",       default="checkpoints")
    p.add_argument("--workers",    type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")
    print(f"Loss mode:  {' + '.join(args.loss)}")

    # Build per-loss weight dict, filling in sensible defaults for any not specified
    _default_weights = {
        "curl_div":    0.0002,
        "spectral":    0.0002,
        "okubo_weiss": 0.001,
        "wasserstein": 1.0,
    }
    aux_losses = [lt for lt in args.loss if lt != "eps"]
    if args.weights is not None:
        if len(args.weights) != len(aux_losses):
            raise ValueError(
                f"--weights has {len(args.weights)} values but "
                f"--loss has {len(aux_losses)} non-eps entries."
            )
        weights = dict(zip(aux_losses, args.weights))
    else:
        weights = {lt: _default_weights.get(lt, 1.0) for lt in aux_losses}

    if aux_losses:
        for lt, w in weights.items():
            print(f"  λ({lt}) = {w}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    land_mask = train_ds.land_mask.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model + diffusion ----
    model = UNet(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    diffusion = MultiLossDDPM(
        T=args.T,
        beta_schedule=args.schedule,
        device=device,
        loss_types=args.loss,
        weights=weights,
        sinkhorn_blur=args.sinkhorn_blur,
    )

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Run tag: model_loss_noise_type_schedule ----
    run_tag = f"ddpm_{'+'.join(args.loss)}_{args.noise_type}_{args.schedule}"

    # ---- Training loop ----
    best_val = float("inf")
    aux_losses = [lt for lt in args.loss if lt != "eps"]

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_total = train_eps = 0.0
        train_indiv = {lt: 0.0 for lt in aux_losses}
        for x0 in train_loader:
            x0 = x0.to(device)
            loss, eps_loss, indiv = diffusion.training_loss(model, x0, land_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item()
            train_eps   += eps_loss.item()
            for lt, v in indiv.items():
                train_indiv[lt] += v.item()
        n = len(train_loader)
        train_total /= n;  train_eps /= n
        for lt in train_indiv: train_indiv[lt] /= n

        # -- Validate --
        model.eval()
        val_total = val_eps = 0.0
        val_indiv = {lt: 0.0 for lt in aux_losses}
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                loss, eps_loss, indiv = diffusion.training_loss(model, x0, land_mask)
                val_total += loss.item()
                val_eps   += eps_loss.item()
                for lt, v in indiv.items():
                    val_indiv[lt] += v.item()
        n = len(val_loader)
        val_total /= n;  val_eps /= n
        for lt in val_indiv: val_indiv[lt] /= n

        scheduler.step()

        # -- Checkpoint --
        saved_best = False
        if val_total < best_val:
            best_val = val_total
            saved_best = True
            torch.save(
                {
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "val_loss":   val_total,
                    "val_eps":    val_eps,
                    "val_indiv":  val_indiv,
                    "args":       vars(args),
                },
                os.path.join(args.save_dir, f"best_{run_tag}.pt"),
            )

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                os.path.join(args.save_dir, f"ckpt_ep{epoch:04d}_{run_tag}.pt"),
            )

        if epoch % 10 == 0 or saved_best:
            tag = " *" if saved_best else ""
            aux_str = "  ".join(f"{lt}={train_indiv[lt]:.5f}" for lt in aux_losses)
            aux_val_str = "  ".join(f"{lt}={val_indiv[lt]:.5f}" for lt in aux_losses)
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={train_total:.5f} (eps={train_eps:.5f}  {aux_str}) | "
                f"val={val_total:.5f}   (eps={val_eps:.5f}  {aux_val_str}){tag}"
            )

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint saved to: {args.save_dir}/best_{run_tag}.pt")


if __name__ == "__main__":
    main()
