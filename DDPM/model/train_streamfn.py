"""
Training script for the STREAM-FUNCTION divergence-free DDPM.

This is a self-contained variation, separate from train.py.  It trains a
StreamFunctionUNet that predicts the clean field x̂₀ as the curl of a scalar
stream function — divergence-free by construction — using a Min-SNR-γ–weighted
reconstruction loss plus a directional (angle) term:

        L = w_t · ‖x̂₀ − x₀‖²_ocean  +  λ · (1 − cosθ)_ocean
        w_t = min(SNR_t, γ) / mean(min(SNR_t, γ))            (Min-SNR-γ, Hang 2023)

Why this design (eddy-detection north star):
  • Stream-function curl head  → output is EXACTLY incompressible (coherent eddies,
    no Leray projection ever needed).
  • x0-prediction              → a CALIBRATED denoiser → valid iterative sampling
    and gradient guidance (DPS), unlike the magnitude-blind angle-only model.
  • Min-SNR-γ weighting        → recovers the noise-level balancing that motivates
    v-prediction while keeping the stream-function (x0-space) parameterisation.
  • Angle term (λ)             → keeps flow DIRECTION first-class.
  • EMA weights                → standard diffusion best practice for stable samples.

Recommended (matches the angle pipeline's data handling):
    python train_streamfn.py \
        --pickle    /path/to/data_divfree.pickle \
        --std_only  --noise_type div_free --schedule cosine \
        --epochs 300 --batch 8 --lr 2e-4 \
        --lambda_angle 1.0 --min_snr_gamma 5.0 \
        --save_dir checkpoints_streamfn
"""

import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader

from dataset   import OceanCurrentDataset
from diffusion import DDPM, NOISE_TYPES
from model     import StreamFunctionUNet


# ---------------------------------------------------------------------------
# Exponential moving average of model weights
# ---------------------------------------------------------------------------

class EMA:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",   default="data.pickle")
    p.add_argument("--epochs",   type=int,   default=300)
    p.add_argument("--batch",    type=int,   default=8)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--base_ch",  type=int,   default=64)
    p.add_argument("--time_dim", type=int,   default=256)
    p.add_argument("--T",        type=int,   default=1000)
    p.add_argument("--noise_type", default="div_free", choices=list(NOISE_TYPES),
                   help="Forward-process noise (default: div_free, matching the "
                        "incompressible stream-function prior).")
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--lambda_angle",  type=float, default=1.0,
                   help="Weight λ on the directional (1−cosθ) term.")
    p.add_argument("--min_snr_gamma", type=float, default=5.0,
                   help="Min-SNR-γ clamp on the reconstruction-loss weight (paper default 5).")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay for the saved weights (0 disables EMA).")
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--save_dir", default="checkpoints_streamfn")
    p.add_argument("--workers",  type=int, default=0)
    p.add_argument("--spectral_filter", default=None,
                   help="Path to spectral_filter.npy for colored div-free noise.")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize data to unit std (mean+std) before training.")
    p.add_argument("--std_only", action="store_true",
                   help="Angle-preserving normalization: divide by std, mean forced "
                        "to 0 (recommended so directions are never rotated).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loss:   Min-SNR x0 (γ={args.min_snr_gamma}) + {args.lambda_angle}·angle")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data (angle-preserving std-only normalization recommended) ----
    if args.std_only:
        _, data_std = OceanCurrentDataset.compute_stats(args.pickle, split=0)
        data_mean = 0.0
        print(f"Std-only normalization (angle-preserving): mean=0.0  std={data_std:.5f}")
    elif args.normalize:
        data_mean, data_std = OceanCurrentDataset.compute_stats(args.pickle, split=0)
        print(f"Normalizing data: mean={data_mean:.5f}  std={data_std:.5f}")
    else:
        data_mean = data_std = None

    train_ds = OceanCurrentDataset(args.pickle, split=0,
                                   data_mean=data_mean, data_std=data_std)
    val_ds   = OceanCurrentDataset(args.pickle, split=1,
                                   data_mean=data_mean, data_std=data_std)

    land_mask = train_ds.land_mask.to(device)   # (H, W) bool

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model + diffusion ----
    model = StreamFunctionUNet(
        in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ---- Optional spectral filter ----
    spec_filter_tensor = None
    if args.spectral_filter:
        import numpy as np
        spec_filter_tensor = torch.from_numpy(
            np.load(args.spectral_filter).astype(np.float32)
        )
        print(f"Spectral filter: {args.spectral_filter}  "
              f"shape={tuple(spec_filter_tensor.shape)}")

    diffusion = DDPM(
        T=args.T,
        beta_schedule=args.schedule,
        device=device,
        noise_type=args.noise_type,
        spectral_filter=spec_filter_tensor,
        noise_scale=args.noise_scale,
    )
    print(f"Noise: {args.noise_type}  scale={args.noise_scale}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    run_tag = f"streamfn_minsnr{args.min_snr_gamma:g}_ang{args.lambda_angle:g}_" \
              f"{args.noise_type}_{args.schedule}"

    def run_epoch(loader, train: bool):
        model.train(train)
        tot = recon = ang = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x0 in loader:
                x0 = x0.to(device)
                loss, recon_mse, indiv = diffusion.training_loss_streamfn(
                    model, x0, land_mask,
                    lambda_angle=args.lambda_angle,
                    min_snr_gamma=args.min_snr_gamma,
                )
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if ema is not None:
                        ema.update(model)
                tot   += loss.item()
                recon += recon_mse.item()
                ang   += indiv["angle"].item()
        n = max(len(loader), 1)
        return tot / n, recon / n, ang / n

    def save_ckpt(path, epoch, val_loss):
        # Save EMA weights (the ones used for sampling) when EMA is on.
        weights = ema.state_dict() if ema is not None else model.state_dict()
        torch.save(
            {"epoch": epoch,
             "model": weights,
             "raw_model": model.state_dict(),
             "pred_type": "x0_streamfn",
             "lambda_angle": args.lambda_angle,
             "min_snr_gamma": args.min_snr_gamma,
             "val_loss": val_loss,
             "args": vars(args),
             "spectral_filter": diffusion.spectral_filter,
             "data_mean": data_mean, "data_std": data_std},
            path,
        )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_tot, tr_recon, tr_ang = run_epoch(train_loader, train=True)
        va_tot, va_recon, va_ang = run_epoch(val_loader,   train=False)
        scheduler.step()

        saved_best = False
        if va_tot < best_val:
            best_val   = va_tot
            saved_best = True
            save_ckpt(os.path.join(args.save_dir, f"best_{run_tag}.pt"),
                      epoch, va_tot)

        if epoch % 10 == 0:
            save_ckpt(os.path.join(args.save_dir, f"ckpt_ep{epoch:04d}_{run_tag}.pt"),
                      epoch, va_tot)

        if epoch % 10 == 0 or saved_best:
            tag = " *" if saved_best else ""
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={tr_tot:.5f} (recon={tr_recon:.5f} ang={tr_ang:.5f}) | "
                f"val={va_tot:.5f}   (recon={va_recon:.5f} ang={va_ang:.5f}){tag}"
            )

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint saved to: {args.save_dir}/best_{run_tag}.pt")


if __name__ == "__main__":
    main()
