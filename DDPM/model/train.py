"""
Training script for the DDPM on ocean current fields.

Supports any combination of structural auxiliary losses via --loss.
Defaults to pure epsilon-MSE (equivalent to the original Basic DDPM).

Usage:
    py train.py
    py train.py --epochs 200 --loss spectral --weights 0.0002
    py train.py --loss spectral okubo_weiss --weights 0.0002 0.001
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset  import OceanCurrentDataset
from diffusion import DDPM, LOSS_MODES, NOISE_TYPES, DEFAULT_WEIGHTS
from model    import UNet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",   default="data.pickle")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--base_ch",  type=int,   default=64)
    p.add_argument("--time_dim", type=int,   default=256)
    p.add_argument("--T",        type=int,   default=1000)
    p.add_argument("--noise_type", default="gaussian", choices=list(NOISE_TYPES),
                   help="Noise type for the forward process: 'gaussian' (default) or "
                        "'div_free' (divergence-free via Fourier projection)")
    p.add_argument("--schedule",   default="cosine", choices=["cosine", "linear"],
                   help="Noise schedule (default: cosine)")
    p.add_argument("--loss",    default=["eps"], choices=LOSS_MODES, nargs="+",
                   help="One or more loss modes (default: eps = plain MSE)")
    p.add_argument("--weights", type=float, default=None, nargs="+",
                   help="Per-loss weights, one per non-eps entry in --loss "
                        "(in the same order). Omit to use defaults.")
    p.add_argument("--sinkhorn_blur", type=float, default=0.05)
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="Std dev of the Gaussian noise in the forward process. "
                        "Set to ~0.12 to match the data scale (default: 1.0 = standard DDPM).")
    p.add_argument("--save_dir", default="models")
    p.add_argument("--workers",  type=int,   default=0)
    p.add_argument("--spectral_filter", default=None,
                   help="Path to spectral_filter.npy for colored div-free noise. "
                        "Only used when --noise_type div_free.")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize data to unit std (ocean cells) before training. "
                        "Stats are saved in the checkpoint for inference.")
    p.add_argument("--std_only", action="store_true",
                   help="Angle-preserving normalization: divide by std but do NOT "
                        "subtract the mean (mean forced to 0). Use this for the angle "
                        "loss so vector directions are not rotated. Takes precedence "
                        "over --normalize. Stats are saved in the checkpoint.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loss:   {' + '.join(args.loss)}")

    # Build per-loss weights dict
    aux_losses = [lt for lt in args.loss if lt != "eps"]
    if args.weights is not None:
        if len(args.weights) != len(aux_losses):
            raise ValueError(
                f"--weights has {len(args.weights)} values but "
                f"--loss has {len(aux_losses)} non-eps entries."
            )
        weights = dict(zip(aux_losses, args.weights))
    else:
        weights = {lt: DEFAULT_WEIGHTS.get(lt, 1.0) for lt in aux_losses}

    for lt, w in weights.items():
        print(f"  \u03bb({lt}) = {w}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data ----
    if args.std_only:
        # Angle-preserving: divide by std only, mean forced to 0 so directions
        # are never rotated (uniform scaling preserves vector angles exactly).
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
    model = UNet(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- Load optional spectral filter ----
    spec_filter_tensor = None
    if args.spectral_filter:
        import numpy as np
        spec_filter_tensor = torch.from_numpy(
            np.load(args.spectral_filter).astype(np.float32)
        )
        print(f"Spectral filter: {args.spectral_filter}  shape={tuple(spec_filter_tensor.shape)}")

    diffusion = DDPM(
        T=args.T,
        beta_schedule=args.schedule,
        device=device,
        noise_type=args.noise_type,
        loss_types=args.loss,
        weights=weights,
        sinkhorn_blur=args.sinkhorn_blur,
        spectral_filter=spec_filter_tensor,
        noise_scale=args.noise_scale,
    )
    print(f"Noise scale: {args.noise_scale}")

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Run tag: ddpm_{losses}_{noise_type}_{schedule} ----
    ns_tag  = f"_ns{args.noise_scale:.2f}".replace(".", "p") if args.noise_scale != 1.0 else ""
    run_tag = f"ddpm_{'+'.join(args.loss)}_{args.noise_type}_{args.schedule}{ns_tag}"

    # ---- Training loop ----
    best_val  = float("inf")

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
                {"epoch": epoch, "model": model.state_dict(),
                 "val_loss": val_total, "val_eps": val_eps,
                 "val_indiv": val_indiv, "args": vars(args),
                 "spectral_filter": diffusion.spectral_filter,
                 "data_mean": data_mean, "data_std": data_std},
                os.path.join(args.save_dir, f"best_{run_tag}.pt"),
            )

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args),
                 "spectral_filter": diffusion.spectral_filter,
                 "data_mean": data_mean, "data_std": data_std},
                os.path.join(args.save_dir, f"ckpt_ep{epoch:04d}_{run_tag}.pt"),
            )

        if epoch % 10 == 0 or saved_best:
            tag = " *" if saved_best else ""
            aux_str     = "  ".join(f"{lt}={train_indiv[lt]:.5f}" for lt in aux_losses)
            aux_val_str = "  ".join(f"{lt}={val_indiv[lt]:.5f}"   for lt in aux_losses)
            aux_part     = f"  {aux_str}"     if aux_str     else ""
            aux_val_part = f"  {aux_val_str}" if aux_val_str else ""
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={train_total:.5f} (eps={train_eps:.5f}{aux_part}) | "
                f"val={val_total:.5f}   (eps={val_eps:.5f}{aux_val_part}){tag}"
            )

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint saved to: {args.save_dir}/best_{run_tag}.pt")


if __name__ == "__main__":
    main()
