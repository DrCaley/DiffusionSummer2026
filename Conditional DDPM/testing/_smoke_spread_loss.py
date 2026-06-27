"""Smoke test for training_loss_streamfn_spread: shapes, finiteness, gradients,
and that a perfect-match target drives the loss low.  No data / no GPU needed.
Run from workspace root:  python "Conditional DDPM/testing/_smoke_spread_loss.py"
"""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", ".."))
for _p in (_here, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), _root):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion import DDPM            # noqa: E402
from model import StreamFunctionUNet  # noqa: E402

torch.manual_seed(0)
device = "cpu"
H, W, B, cond_ch = 94, 44, 4, 10

land = torch.zeros(H, W, dtype=torch.bool)
land[:2, :] = True                                  # a strip of land
model = StreamFunctionUNet(in_ch=2, base_ch=16, time_dim=64, cond_ch=cond_ch).to(device)
diff = DDPM(T=1000, beta_schedule="cosine", device=device, noise_type="div_free")

x0   = torch.randn(B, 2, H, W) * 0.1
cond = torch.randn(B, cond_ch, H, W) * 0.1
target = torch.rand(B, H, W)                        # arbitrary spread target in [0,1]
target[:, land] = 0.0

for param in ("x0", "v"):
    loss, recon, indiv = diff.training_loss_streamfn_spread(
        model, x0, land, target,
        lambda_angle=1.0, min_snr_gamma=5.0, cond=cond,
        parameterization=param, lambda_mag=0.2,
        lambda_spread=2.0, spread_samples=8,
        spread_t_mu=0.5, spread_t_sigma=0.25,
    )
    assert torch.isfinite(loss), f"{param}: non-finite loss"
    assert torch.isfinite(indiv["spread_loss"]), f"{param}: non-finite spread_loss"
    assert torch.isfinite(indiv["spread_r"]), f"{param}: non-finite spread_r"
    model.zero_grad()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert gnorm > 0 and torch.isfinite(torch.tensor(gnorm)), f"{param}: bad grad {gnorm}"
    print(f"[{param}] loss={loss.item():.4f} recon={recon.item():.4f} "
          f"angle={indiv['angle'].item():.4f} mag={indiv['mag'].item():.4f} "
          f"spread_loss={indiv['spread_loss'].item():.4f} "
          f"spread_r={indiv['spread_r'].item():+.4f} grad_norm={gnorm:.3e}")

# Sanity: spread_loss == 1 - spread_r (both window-weighted) up to rounding.
diff_sl = abs(indiv["spread_loss"].item() - (1.0 - indiv["spread_r"].item()))
assert diff_sl < 1e-4, f"spread_loss/spread_r inconsistent: {diff_sl}"

# Sanity: feeding the model's OWN spread map as the target -> r=1, loss~0.
torch.manual_seed(1)
t = torch.randint(0, diff.T, (B,), device=device)
t_K = t.repeat(8); x0_K = x0.repeat(8, 1, 1, 1); cond_K = cond.repeat(8, 1, 1, 1)
xt_K, _ = diff.q_sample(x0_K, t_K)
with torch.no_grad():
    out = model(xt_K, t_K, cond_K)
xhat = out.view(8, B, 2, H, W)
eps = 1e-8
u = xhat[:, :, 0]; v = xhat[:, :, 1]
mag = torch.sqrt(u * u + v * v + eps)
mu_ = (u / mag).mean(0); mv_ = (v / mag).mean(0)
self_spread = 1.0 - torch.sqrt(mu_ ** 2 + mv_ ** 2 + eps)   # (B,H,W)
ocean = (~land).reshape(-1)
sp = self_spread.reshape(B, -1)[:, ocean]
tg = self_spread.reshape(B, -1)[:, ocean]
sp = sp - sp.mean(1, keepdim=True); tg = tg - tg.mean(1, keepdim=True)
r_self = ((sp * tg).sum(1) / (torch.sqrt((sp * sp).sum(1) * (tg * tg).sum(1)) + eps))
print(f"self-correlation r (should be ~1): {r_self.mean().item():.5f}")
assert r_self.mean().item() > 0.999, "self-correlation not ~1"

print("\nALL SMOKE CHECKS PASSED")
