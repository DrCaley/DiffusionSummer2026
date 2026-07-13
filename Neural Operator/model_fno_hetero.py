"""
model_fno_hetero.py
=====================
Heteroscedastic single-shot FNO for ocean-current reconstruction.

Unlike every other model in this project, this one is NOT a diffusion
denoiser — there's no time embedding and no iterative sampling. It takes the
sparse path observation directly as input and predicts the full field in one
forward pass:

    FNOHetero(x0_obs, path_mask, ocean_mask) -> (mean, log_var)

Non-determinism comes from sampling x0 ~ N(mean, exp(log_var)) rather than
from a multi-step reverse-diffusion trajectory starting at pure noise. Since
the network is trained end-to-end for exactly this reconstruction task (not
generic denoising), the mean prediction alone should already be close to the
task's achievable floor — and because sigma is *learned* per-pixel, it can be
near-zero close to observations and larger far from them, so sampling adds
only as much randomness as the model is actually uncertain about.

Input channels: x0_obs (2: u, v; zero outside the path) + path_mask (1) +
ocean_mask (1) = 4. No spatial padding needed — FNO operates natively at
(94, 44) since there's no strided downsampling in this architecture.
"""

import os
import sys

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repaint_model_dir():
    candidates = [
        os.path.join(_SCRIPT_DIR, "..", "Repaint vs DPS"),
        os.path.join(_SCRIPT_DIR, "..", "Repaint_vs_DPS"),
        "/root/Repaint_vs_DPS",
    ]
    for d in candidates:
        d = os.path.abspath(d)
        if os.path.isfile(os.path.join(d, "repaint_model.py")):
            return d
    raise RuntimeError("Cannot find repaint_model.py — tried: " + str(candidates))


_rp_dir = _find_repaint_model_dir()
if _rp_dir not in sys.path:
    sys.path.insert(0, _rp_dir)
from repaint_model import _num_groups  # noqa: E402

from model_fno_ddpm import SpectralConv2d  # noqa: E402


class PlainFNOBlock(nn.Module):
    """Non-time-conditioned spectral residual block (same shape as the one
    in model_fno_autoencoder.py — duplicated here to keep this file
    self-contained aside from the two small shared imports above)."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.norm  = nn.GroupNorm(_num_groups(width), width)
        self.spec  = SpectralConv2d(width, width, modes1, modes2)
        self.point = nn.Conv2d(width, width, 1)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(x))
        h = self.act(self.spec(h) + self.point(h))
        return x + h


class FNOHetero(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 2, width: int = 64,
                 modes1: int = 16, modes2: int = 16, n_layers: int = 4,
                 logvar_init: float = -2.0):
        super().__init__()
        self.fc0    = nn.Linear(in_ch, width)
        self.blocks = nn.ModuleList([
            PlainFNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])
        self.fc1        = nn.Linear(width, 128)
        self.fc_mean    = nn.Linear(128, out_ch)
        self.fc_logvar  = nn.Linear(128, out_ch)
        self.act = nn.GELU()

        nn.init.constant_(self.fc_logvar.bias, logvar_init)
        nn.init.zeros_(self.fc_logvar.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, in_ch, H, W) — [x0_obs(2), path_mask(1), ocean_mask(1)]
        Returns:
            mean:    (B, out_ch, H, W)
            log_var: (B, out_ch, H, W)
        """
        h = x.permute(0, 2, 3, 1)              # (B, H, W, in_ch)
        h = self.fc0(h)
        h = h.permute(0, 3, 1, 2).contiguous()  # (B, width, H, W)

        for block in self.blocks:
            h = block(h)

        h = h.permute(0, 2, 3, 1)               # (B, H, W, width)
        h = self.act(self.fc1(h))
        mean    = self.fc_mean(h).permute(0, 3, 1, 2).contiguous()
        log_var = self.fc_logvar(h).permute(0, 3, 1, 2).contiguous()
        log_var = log_var.clamp(-10.0, 4.0)     # keep exp() numerically stable
        return mean, log_var


def build_input(x0_obs: torch.Tensor, path_mask: torch.Tensor,
                ocean_mask: torch.Tensor) -> torch.Tensor:
    """
    x0_obs:     (B, 2, H, W)
    path_mask:  (B, 1, H, W) or (1, 1, H, W) float, 1 = known
    ocean_mask: (B, 1, H, W) or (1, 1, H, W) float, 1 = ocean
    Returns (B, 4, H, W).
    """
    B = x0_obs.shape[0]
    if path_mask.shape[0] == 1 and B > 1:
        path_mask = path_mask.expand(B, -1, -1, -1)
    if ocean_mask.shape[0] == 1 and B > 1:
        ocean_mask = ocean_mask.expand(B, -1, -1, -1)
    return torch.cat([x0_obs, path_mask, ocean_mask], dim=1)
