"""
Physics-Informed Diffusion Training Script
==========================================
Based on: "A Physics-informed Diffusion Model for High-fidelity Flow Field
Reconstruction" (Shu, Li, Farimani – arXiv:2211.14680, JCP 2023).

Key additions over the baseline train.py:
  1. Divergence-free (continuity) residual loss:
       L_div = mean(div(x0_hat)^2) on ocean pixels
     Forces the denoised prediction to respect mass conservation.

  2. Helmholtz stream-function loss:
       L_stream = ||u_hat - d(psi)/dy||^2 + ||v_hat - (-d(psi)/dx)||^2
     where psi is recovered via a fast Poisson solve (FFT-based).
     Encourages the model to produce fields that are exactly representable
     by a stream function (2-D incompressible flow constraint).

  3. Okubo-Weiss structural loss:
       L_ow = MSE between Okubo-Weiss parameter of x0_hat and x0_true
     Preserves eddy / strain-rate structure.

  4. Combined loss:
       L = L_eps + w_div * L_div + w_stream * L_stream + w_ow * L_ow

Loss weights (tuneable via CLI):
  --div_weight     0.05   (divergence / continuity)
  --stream_weight  0.01   (stream-function reconstruction)
  --ow_weight      0.001  (Okubo-Weiss structure)

Training stops when val loss does not improve for --patience epochs (default 50).

Usage (on Vast.ai):
    cd /root/stjohn_ddpm
    nohup python3 train_physics_informed.py \\
        --pickle data.pickle \\
        --schedule linear \\
        --epochs 2000 \\
        --patience 50 \\
        --div_weight 0.05 \\
        --stream_weight 0.01 \\
        --ow_weight 0.001 \\
        --save_dir /root/stjohn_ddpm/checkpoints_physics \\
        > /root/stjohn_ddpm/train_physics.log 2>&1 &
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import OceanCurrentDataset
from diffusion import DDPM
from repaint_model import Repaint


# ---------------------------------------------------------------------------
# Finite-difference helpers (shared with loss_functions.py but self-contained)
# ---------------------------------------------------------------------------

def _jacobian(field: torch.Tensor):
    """Return (du_dx, du_dy, dv_dx, dv_dy) via central differences."""
    kx = torch.tensor([[[[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]]]], device=field.device)
    ky = torch.tensor([[[[0., -0.5, 0.], [0., 0., 0.], [0., 0.5, 0.]]]], device=field.device)
    u = field[:, :1]
    v = field[:, 1:]
    return (
        F.conv2d(u, kx, padding=1),
        F.conv2d(u, ky, padding=1),
        F.conv2d(v, kx, padding=1),
        F.conv2d(v, ky, padding=1),
    )


def divergence_loss(pred: torch.Tensor, ocean: torch.Tensor) -> torch.Tensor:
    """
    Mean squared divergence of pred on ocean pixels.
    pred:  (B, 2, H, W)
    ocean: (1, 1, H, W) float mask
    """
    du_dx, _, dv_dx, dv_dy = _jacobian(pred)
    div = (du_dx + dv_dy) * ocean
    return (div ** 2).mean()


def okubo_weiss_loss(pred: torch.Tensor, true: torch.Tensor, ocean: torch.Tensor) -> torch.Tensor:
    """
    MSE between the Okubo-Weiss parameter W of pred and true.
    W = Sn^2 + Ss^2 - omega^2
      Sn = du/dx - dv/dy  (normal strain)
      Ss = dv/dx + du/dy  (shear strain)
      omega = dv/dx - du/dy  (vorticity)
    """
    def _ow(field):
        du_dx, du_dy, dv_dx, dv_dy = _jacobian(field)
        Sn = du_dx - dv_dy
        Ss = dv_dx + du_dy
        omega = dv_dx - du_dy
        return Sn ** 2 + Ss ** 2 - omega ** 2

    W_pred = _ow(pred) * ocean
    W_true = _ow(true) * ocean
    return F.mse_loss(W_pred, W_true)


def _fft_poisson_solve(rhs: torch.Tensor) -> torch.Tensor:
    """
    Solve Laplacian(phi) = rhs via FFT on a 2-D periodic domain.
    rhs: (B, 1, H, W)
    returns phi: same shape
    """
    B, C, H, W = rhs.shape
    rhs_hat = torch.fft.rfft2(rhs)
    ky = torch.fft.fftfreq(H, d=1.0 / (2 * torch.pi), device=rhs.device)
    kx = torch.fft.rfftfreq(W, d=1.0 / (2 * torch.pi), device=rhs.device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")  # (H, W//2+1)
    denom = -(KX ** 2 + KY ** 2)
    denom[0, 0] = 1.0  # avoid divide-by-zero; DC component set to 0
    phi_hat = rhs_hat / denom.unsqueeze(0).unsqueeze(0)
    phi_hat[:, :, 0, 0] = 0.0  # zero mean
    return torch.fft.irfft2(phi_hat, s=(H, W))


def stream_function_loss(pred: torch.Tensor, ocean: torch.Tensor) -> torch.Tensor:
    """
    Encourage the field to be derivable from a stream function psi:
      u =  d(psi)/dy,   v = -d(psi)/dx
    Recover psi by solving  Laplacian(psi) = vorticity = dv/dx - du/dy.
    Then penalise reconstruction error:
      L = ||u - d(psi)/dy||^2 + ||v + d(psi)/dx||^2   (ocean only)
    """
    _, du_dy, dv_dx, _ = _jacobian(pred)
    vorticity = dv_dx - du_dy                       # (B, 1, H, W)
    psi = _fft_poisson_solve(vorticity)              # (B, 1, H, W)

    kx = torch.tensor([[[[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]]]], device=pred.device)
    ky = torch.tensor([[[[0., -0.5, 0.], [0., 0., 0.], [0., 0.5, 0.]]]], device=pred.device)
    dpsi_dx = F.conv2d(psi, kx, padding=1)
    dpsi_dy = F.conv2d(psi, ky, padding=1)

    u_rec = dpsi_dy          # u =  d(psi)/dy
    v_rec = -dpsi_dx         # v = -d(psi)/dx

    diff_u = ((pred[:, :1] - u_rec) * ocean) ** 2
    diff_v = ((pred[:, 1:] - v_rec) * ocean) ** 2
    return (diff_u + diff_v).mean()


# ---------------------------------------------------------------------------
# Physics-Informed DDPM training loss
# ---------------------------------------------------------------------------

class PhysicsInformedDDPM(DDPM):
    """
    Extends DDPM.training_loss with three additive physics terms:
      div_weight    * divergence_loss(x0_hat)
      stream_weight * stream_function_loss(x0_hat)
      ow_weight     * okubo_weiss_loss(x0_hat, x0)
    """

    def __init__(self, *args,
                 div_weight: float = 0.05,
                 stream_weight: float = 0.01,
                 ow_weight: float = 0.001,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.div_weight    = div_weight
        self.stream_weight = stream_weight
        self.ow_weight     = ow_weight

    def training_loss(self, model, x0, land_mask, **kwargs):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)

        ocean = (~land_mask).float()[None, None]   # (1,1,H,W)

        # Base epsilon-MSE
        eps_loss = F.mse_loss(pred_noise * ocean, noise * ocean)

        # Recover denoised estimate x0_hat
        ab = self.alpha_bar[t][:, None, None, None]
        x0_hat = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        # Physics losses
        div_loss = divergence_loss(x0_hat, ocean) if self.div_weight > 0 else torch.tensor(0.0, device=self.device)
        sf_loss  = stream_function_loss(x0_hat, ocean) if self.stream_weight > 0 else torch.tensor(0.0, device=self.device)
        ow_loss  = okubo_weiss_loss(x0_hat, x0, ocean) if self.ow_weight > 0 else torch.tensor(0.0, device=self.device)

        total = (eps_loss
                 + self.div_weight    * div_loss
                 + self.stream_weight * sf_loss
                 + self.ow_weight     * ow_loss)
        return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Physics-Informed DDPM training for ocean currents.")
    p.add_argument("--pickle",        default="data.pickle")
    p.add_argument("--schedule",      default="linear",
                   choices=["linear", "cosine", "geometric", "quadratic", "sigmoid"])
    p.add_argument("--epochs",        type=int,   default=2000)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--base_ch",       type=int,   default=64)
    p.add_argument("--time_dim",      type=int,   default=256)
    p.add_argument("--T",             type=int,   default=1000)
    p.add_argument("--patience",      type=int,   default=50)
    p.add_argument("--save_dir",      default=None)
    p.add_argument("--resume",        default=None)
    p.add_argument("--div_weight",    type=float, default=0.05,
                   help="Weight for divergence (continuity) residual loss.")
    p.add_argument("--stream_weight", type=float, default=0.01,
                   help="Weight for stream-function reconstruction loss.")
    p.add_argument("--ow_weight",     type=float, default=0.001,
                   help="Weight for Okubo-Weiss structural loss.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.save_dir is None:
        args.save_dir = os.path.join(script_dir, "checkpoints_physics")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device          : {device}")
    print(f"Schedule        : {args.schedule}")
    print(f"Save dir        : {args.save_dir}")
    print(f"div_weight      : {args.div_weight}")
    print(f"stream_weight   : {args.stream_weight}")
    print(f"ow_weight       : {args.ow_weight}")

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)
    land_mask = train_ds.land_mask.to(device)    # (H, W) bool

    noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())
    print(f"noise_std       : {noise_std:.5f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # ---- Model ----
    model = Repaint(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    print(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}")

    # ---- Physics-Informed diffusion ----
    diffusion = PhysicsInformedDDPM(
        T=args.T, beta_schedule=args.schedule, device=device, noise_std=noise_std,
        curl_div_weight=0.002,       # keep existing structural term
        div_weight=args.div_weight,
        stream_weight=args.stream_weight,
        ow_weight=args.ow_weight,
    )

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Resume ----
    start_epoch  = 0
    best_val     = float("inf")
    patience_ctr = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        if "val_loss"  in ckpt: best_val = ckpt["val_loss"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.5f}")

    def _get_x0(batch, device):
        """Support datasets returning plain tensors or (x0, *cond) tuples/lists."""
        if isinstance(batch, (list, tuple)):
            return batch[0].to(device)
        return batch.to(device)

    # ---- Training loop ----
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x0 = _get_x0(batch, device)
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
            for batch in val_loader:
                x0 = _get_x0(batch, device)
                val_loss += diffusion.training_loss(model, x0, land_mask).item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        ckpt_data = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "val_loss":      val_loss,
            "noise_std":     noise_std,
            "schedule":      args.schedule,
            "div_weight":    args.div_weight,
            "stream_weight": args.stream_weight,
            "ow_weight":     args.ow_weight,
            "args":          vars(args),
        }

        if val_loss < best_val:
            best_val     = val_loss
            patience_ctr = 0
            torch.save(ckpt_data, os.path.join(args.save_dir, "best_model_physics.pt"))
        else:
            patience_ctr += 1

        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint : {args.save_dir}/best_model_physics.pt")


if __name__ == "__main__":
    main()
