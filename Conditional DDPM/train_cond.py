"""
Training script for the CONDITIONAL stream-function divergence-free DDPM.

This is the conditional counterpart of ``train_streamfn.py``.  It trains a
``StreamFunctionUNet`` whose first convolution is widened to ingest, alongside
the noisy field x_t, a stack of conditioning channels (soft robot-path
observations, temporal-prior fields, and static geometry — see
``cond_dataset``).  Everything downstream is unchanged: the network still emits
a single scalar stream function ψ whose curl gives an EXACTLY divergence-free
(u, v) field.  By default it is trained with **v-prediction** (Min-SNR-γ weighted
MSE on the velocity target) — the well-conditioned parameterisation that keeps
magnitude balanced at all noise levels while down-weighting the highest-noise
steps that otherwise drive full-magnitude high-frequency speckle in curl(ψ) —
plus the directional angle loss:

    input  = [ x_t (2) | obs (3) | priors (2·|lags|) | geom (3) ]  → ψ → curl → (u,v)
    v      = √ᾱ·ε − √(1−ᾱ)·x₀ ,   x̂₀ = √ᾱ·x_t − √(1−ᾱ)·v̂
    L      = w_t · ‖v̂ − v‖²_ocean  +  λ · (1 − cosθ)_ocean ,
             w_t = min(SNR_t, γ)/(SNR_t + 1)   (Hang et al., 2023; ≤0 ⇒ plain MSE)

(Pass ``--parameterization x0`` to recover the original Min-SNR-γ x0 loss.)

North star: given a few known pixels (the robot path) plus recent history,
produce a PLAUSIBLE, divergence-free, well-calibrated full current field — and,
via the diffusion non-determinism, a diverse ensemble of such guesses.

Recommended (from the workspace root):
    python "Conditional DDPM/train_cond.py" \
        --pickle    Datasets/data_divfree_chrono.pickle \
        --std_only  --noise_type div_free --schedule cosine \
        --parameterization v --min_snr_gamma 5.0 \
        --epochs 300 --batch 8 --lr 2e-4 --patience 50 \
        --lambda_angle 1.0 \
        --lags 13,25 --path_steps 50,400 --workers 4 \
        --save_dir "Conditional DDPM/checkpoints_cond"

The cosine LR schedule decays to a floor of ``--eta_min`` (default lr*0.05, not
0) so the late epochs keep learning instead of annealing to a standstill.
"""

import argparse
import copy
import os
import sys

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup — the shared model/diffusion code lives in DDPM/model and the
# conditional dataset in utils.  Works from the workspace root or a flat layout.
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, ".."))
for _p in [_root, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), _here]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from cond_dataset import ConditionalOceanDataset, cond_channels
from diffusion    import DDPM, NOISE_TYPES
from model        import StreamFunctionUNet


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

def _parse_steps(spec: str):
    """'150' -> 150 ; '50,400' -> (50, 400)."""
    if "," in spec:
        lo, hi = spec.split(",")
        return (int(lo), int(hi))
    return int(spec)


def _parse_lags(spec: str):
    """'13,25' -> (13, 25)."""
    return tuple(int(s) for s in spec.split(","))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",   default="Datasets/data_divfree_chrono.pickle")
    p.add_argument("--epochs",   type=int,   default=300)
    p.add_argument("--batch",    type=int,   default=8)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--eta_min",  type=float, default=None,
                   help="Cosine-schedule LR floor.  None (default) = lr*0.05, so "
                        "the late epochs keep doing real work instead of "
                        "annealing to ~0.  Pass an explicit value (e.g. 0) to "
                        "override.")
    p.add_argument("--base_ch",  type=int,   default=64)
    p.add_argument("--time_dim", type=int,   default=256)
    p.add_argument("--T",        type=int,   default=1000)
    p.add_argument("--noise_type", default="div_free", choices=list(NOISE_TYPES),
                   help="Forward-process noise (default: div_free, matching the "
                        "incompressible stream-function prior).")
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--parameterization", default="v", choices=["x0", "v"],
                   help="Network target: 'v' (v-prediction, default — Min-SNR-γ "
                        "weighted, keeps magnitude balanced while down-weighting "
                        "high-noise speckle pressure) or 'x0' (Min-SNR-weighted "
                        "x0-prediction, the original behaviour).")
    p.add_argument("--lambda_angle",  type=float, default=1.0,
                   help="Weight λ on the directional (1−cosθ) term.")
    p.add_argument("--lambda_mag",    type=float, default=0.0,
                   help="Weight on the energy-matching term L_mag = "
                        "mean[(rms(x̂₀)/rms(x₀)-1)²] over ocean cells.  Fights the "
                        "MSE mean-seeking that shrinks field magnitude (measured "
                        "~87%% at low noise).  0 disables (default).  Try 0.5.")
    p.add_argument("--lambda_vort",   type=float, default=0.0,
                   help="Weight on the vorticity-matching term L_vort = "
                        "mean[(ω(x̂₀)-ω(x₀))²], ω=∂v/∂x-∂u/∂y (central diff).  A "
                        "high-frequency structure loss that fights MSE blur with "
                        "the physically natural scalar for incompressible flow "
                        "(the clean 'curl' half of curl-div).  0 disables "
                        "(default).  Try 0.1.")
    p.add_argument("--lambda_energy", type=float, default=0.0,
                   help="Weight on a strictly-proper ENERGY SCORE (CRPS) that "
                        "supervises the ensemble SPREAD — the only term that "
                        "trains predictive uncertainty (what r_dir measures).  "
                        "Draws --energy_samples predictions per example and "
                        "rewards correct dispersion.  0 disables (default).  "
                        "Try 0.5.  Reduce --batch since each step is K× larger.")
    p.add_argument("--energy_samples", type=int, default=4,
                   help="K, predictive samples per example for the energy score "
                        "(>=2).  Only used when --lambda_energy > 0.")
    p.add_argument("--init", default=None,
                   help="Path to an existing checkpoint to initialise weights "
                        "from (fine-tuning).  Loads the 'model' state dict; the "
                        "optimizer / schedule restart fresh.")
    p.add_argument("--min_snr_gamma", type=float, default=5.0,
                   help="Min-SNR-γ clamp on the loss weight.  x0: w=min(SNR,γ); "
                        "v: w=min(SNR,γ)/(SNR+1).  Set ≤0 for plain v-MSE.")
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop after this many epochs with no val improvement "
                        "(0 disables; the best checkpoint is always kept).")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay for the saved weights (0 disables EMA).")
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--save_dir", default="Conditional DDPM/checkpoints_cond")
    p.add_argument("--workers",  type=int, default=0)
    p.add_argument("--spectral_filter", default=None,
                   help="Path to spectral_filter.npy for colored div-free noise.")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize data to unit std (mean+std) before training.")
    p.add_argument("--std_only", action="store_true",
                   help="Angle-preserving normalization: divide by std, mean forced "
                        "to 0 (recommended so directions are never rotated).")
    # ---- conditioning-specific ----
    p.add_argument("--lags", type=_parse_lags, default=(13, 25),
                   help="Comma-separated temporal-prior lags in hours/frames "
                        "(default 13,25).")
    p.add_argument("--path_steps", type=_parse_steps, default=(120, 200),
                   help="Robot-path length: a fixed int (e.g. 150) or a 'min,max' "
                        "range sampled per sample for modest coverage augmentation "
                        "(default 120,200 — known-cell count varies but not too much).")
    p.add_argument("--straight_bias", type=float, default=0.75,
                   help="Directional-persistence bias for the biased random walk.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    if args.parameterization == "v":
        if args.min_snr_gamma > 0:
            print(f"Loss:   v-prediction (Min-SNR γ={args.min_snr_gamma}) + {args.lambda_angle}·angle")
        else:
            print(f"Loss:   v-prediction (plain MSE) + {args.lambda_angle}·angle")
    else:
        print(f"Loss:   Min-SNR x0 (γ={args.min_snr_gamma}) + {args.lambda_angle}·angle")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Normalization (angle-preserving std-only recommended) ----
    if args.std_only:
        _, data_std = ConditionalOceanDataset.compute_stats(args.pickle, split=0)
        data_mean = 0.0
        print(f"Std-only normalization (angle-preserving): mean=0.0  std={data_std:.5f}")
    elif args.normalize:
        data_mean, data_std = ConditionalOceanDataset.compute_stats(args.pickle, split=0)
        print(f"Normalizing data: mean={data_mean:.5f}  std={data_std:.5f}")
    else:
        data_mean = data_std = None

    # ---- Data ----
    train_ds = ConditionalOceanDataset(
        args.pickle, split=0, lags=args.lags,
        data_mean=data_mean, data_std=data_std,
        path_steps=args.path_steps, deterministic=False,
        straight_bias=args.straight_bias,
    )
    val_ds = ConditionalOceanDataset(
        args.pickle, split=1, lags=args.lags,
        data_mean=data_mean, data_std=data_std,
        path_steps=args.path_steps, deterministic=True,   # reproducible val paths
        straight_bias=args.straight_bias,
    )

    land_mask = train_ds.land_mask.to(device)   # (H, W) bool
    cond_ch   = cond_channels(args.lags)
    print(f"Lags: {args.lags}  path_steps: {args.path_steps}  cond_channels: {cond_ch}")
    print(f"Samples: train={len(train_ds)}  val={len(val_ds)}")

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
        in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim, cond_ch=cond_ch,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- Optional fine-tune init from an existing checkpoint ----
    if args.init:
        ckpt_init = torch.load(args.init, map_location=device, weights_only=False)
        state = ckpt_init.get("raw_model", ckpt_init.get("model"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Fine-tune init from {args.init}  (ep {ckpt_init.get('epoch','?')}, "
              f"val {ckpt_init.get('val_loss', float('nan')):.5f}; "
              f"missing={len(missing)} unexpected={len(unexpected)})")

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
    eta_min = args.eta_min if args.eta_min is not None else args.lr * 0.05
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=eta_min)
    print(f"Schedule: cosine over {args.epochs} epochs, "
          f"lr {args.lr:g} -> eta_min {eta_min:g}")

    lag_tag = "-".join(str(l) for l in args.lags)
    if args.parameterization == "v":
        param_tag = f"vsnr{args.min_snr_gamma:g}" if args.min_snr_gamma > 0 else "v"
    else:
        param_tag = f"minsnr{args.min_snr_gamma:g}"
    if args.lambda_mag > 0:
        param_tag += f"_mag{args.lambda_mag:g}"
    if args.lambda_vort > 0:
        param_tag += f"_vort{args.lambda_vort:g}"
    if args.lambda_energy > 0:
        param_tag += f"_en{args.lambda_energy:g}k{args.energy_samples}"
    run_tag = f"streamfncond_{param_tag}_ang{args.lambda_angle:g}_" \
              f"lags{lag_tag}_{args.noise_type}_{args.schedule}"

    use_energy = args.lambda_energy > 0

    def run_epoch(loader, train: bool):
        model.train(train)
        tot = recon = ang = mag = vort = energy = spread = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                x0   = batch["target"].to(device)
                cond = batch["cond"].to(device)
                if use_energy:
                    loss, recon_mse, indiv = diffusion.training_loss_streamfn_energy(
                        model, x0, land_mask,
                        lambda_angle=args.lambda_angle,
                        min_snr_gamma=args.min_snr_gamma,
                        cond=cond,
                        parameterization=args.parameterization,
                        lambda_mag=args.lambda_mag,
                        lambda_vort=args.lambda_vort,
                        lambda_energy=args.lambda_energy,
                        energy_samples=args.energy_samples,
                    )
                else:
                    loss, recon_mse, indiv = diffusion.training_loss_streamfn(
                        model, x0, land_mask,
                        lambda_angle=args.lambda_angle,
                        min_snr_gamma=args.min_snr_gamma,
                        cond=cond,
                        parameterization=args.parameterization,
                        lambda_mag=args.lambda_mag,
                        lambda_vort=args.lambda_vort,
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
                if "mag" in indiv:
                    mag += indiv["mag"].item()
                if "vort" in indiv:
                    vort += indiv["vort"].item()
                if "energy" in indiv:
                    energy += indiv["energy"].item()
                if "spread" in indiv:
                    spread += indiv["spread"].item()
        n = max(len(loader), 1)
        return (tot / n, recon / n, ang / n, mag / n, vort / n,
                energy / n, spread / n)

    def save_ckpt(path, epoch, val_loss):
        weights = ema.state_dict() if ema is not None else model.state_dict()
        torch.save(
            {"epoch": epoch,
             "model": weights,
             "raw_model": model.state_dict(),
             "pred_type": f"{args.parameterization}_streamfn_cond",
             "parameterization": args.parameterization,
             "lambda_angle": args.lambda_angle,
             "min_snr_gamma": args.min_snr_gamma,
             "lambda_mag": args.lambda_mag,
             "lambda_vort": args.lambda_vort,
             "lambda_energy": args.lambda_energy,
             "energy_samples": args.energy_samples,
             "cond_ch": cond_ch,
             "lags": list(args.lags),
             "path_steps": args.path_steps,
             "val_loss": val_loss,
             "args": vars(args),
             "spectral_filter": diffusion.spectral_filter,
             "data_mean": data_mean, "data_std": data_std},
            path,
        )

    best_val = float("inf")
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        tr_tot, tr_recon, tr_ang, tr_mag, tr_vort, tr_en, tr_spr = run_epoch(train_loader, train=True)
        va_tot, va_recon, va_ang, va_mag, va_vort, va_en, va_spr = run_epoch(val_loader,   train=False)
        scheduler.step()

        saved_best = False
        if va_tot < best_val:
            best_val   = va_tot
            saved_best = True
            epochs_since_best = 0
            save_ckpt(os.path.join(args.save_dir, f"best_{run_tag}.pt"),
                      epoch, va_tot)
        else:
            epochs_since_best += 1

        if epoch % 10 == 0:
            save_ckpt(os.path.join(args.save_dir, f"ckpt_ep{epoch:04d}_{run_tag}.pt"),
                      epoch, va_tot)

        if epoch % 10 == 0 or saved_best:
            tag = " *" if saved_best else ""
            mag_str = f" mag={tr_mag:.5f}/{va_mag:.5f}" if args.lambda_mag > 0 else ""
            vort_str = f" vort={tr_vort:.5f}/{va_vort:.5f}" if args.lambda_vort > 0 else ""
            en_str = (f" en={tr_en:+.5f}/{va_en:+.5f} spr={tr_spr:.5f}/{va_spr:.5f}"
                      if args.lambda_energy > 0 else "")
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={tr_tot:.5f} (recon={tr_recon:.5f} ang={tr_ang:.5f}) | "
                f"val={va_tot:.5f}   (recon={va_recon:.5f} ang={va_ang:.5f}){mag_str}{vort_str}{en_str}{tag}"
            )

        if args.patience > 0 and epochs_since_best >= args.patience:
            print(
                f"\nEarly stopping at epoch {epoch}: no val improvement for "
                f"{args.patience} epochs (best val={best_val:.5f})."
            )
            break

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint saved to: {args.save_dir}/best_{run_tag}.pt")


if __name__ == "__main__":
    main()
