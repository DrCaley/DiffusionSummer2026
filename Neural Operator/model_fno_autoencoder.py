"""
model_fno_autoencoder.py
==========================
FNO-based autoencoder for latent diffusion.

Unlike the original model_fno.py FNO2d (which maps (94,44) -> (94,44) at
constant spatial resolution — a channel-mixing autoencoder with no real
bottleneck), this one has an actual bottleneck: two stride-2 convolutions
compress (2, 94, 44) down to (latent_ch, 24, 11) before a stack of spectral
(FNO) residual blocks, then a mirrored decoder upsamples back.

    Field dims:  2 * 94 * 44  = 8,272
    Latent dims: 8 * 24 * 11  = 2,112   (~3.9x compression)

Trained as a plain reconstruction autoencoder (masked MSE on ocean pixels).
Once trained and frozen, its latent codes are the space a separate latent
DDPM (see model_fno_ddpm.py, reused at the latent resolution) is trained to
denoise — see train_latent_fno_ddpm.py.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# Plain (non-time-conditioned) spectral residual block, for the autoencoder
# ---------------------------------------------------------------------------

class PlainFNOBlock(nn.Module):
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


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

# Pad (94, 44) -> (96, 44): only height needs padding (width is already /4).
_PAD = (0, 0, 1, 1)   # F.pad convention: (left, right, top, bottom)


class FNOEncoder(nn.Module):
    """(B, 2, 94, 44) -> (B, latent_ch, 24, 11)"""

    def __init__(self, in_ch: int = 2, base: int = 32, latent_ch: int = 8,
                 modes1: int = 12, modes2: int = 6, n_blocks: int = 2):
        super().__init__()
        self.down1 = nn.Conv2d(in_ch, base,     4, stride=2, padding=1)  # 96x44 -> 48x22
        self.down2 = nn.Conv2d(base,  base * 2,  4, stride=2, padding=1)  # 48x22 -> 24x11
        self.blocks = nn.ModuleList([
            PlainFNOBlock(base * 2, modes1, modes2) for _ in range(n_blocks)
        ])
        self.to_latent = nn.Conv2d(base * 2, latent_ch, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, _PAD)                # (B, 2, 96, 44)
        h = self.act(self.down1(x))       # (B, base, 48, 22)
        h = self.act(self.down2(h))       # (B, base*2, 24, 11)
        for block in self.blocks:
            h = block(h)
        return self.to_latent(h)          # (B, latent_ch, 24, 11)


class FNODecoder(nn.Module):
    """(B, latent_ch, 24, 11) -> (B, out_ch, 94, 44)"""

    def __init__(self, out_ch: int = 2, base: int = 32, latent_ch: int = 8,
                 modes1: int = 12, modes2: int = 6, n_blocks: int = 2):
        super().__init__()
        self.from_latent = nn.Conv2d(latent_ch, base * 2, 1)
        self.blocks = nn.ModuleList([
            PlainFNOBlock(base * 2, modes1, modes2) for _ in range(n_blocks)
        ])
        self.up1 = nn.ConvTranspose2d(base * 2, base,   4, stride=2, padding=1)  # 24x11 -> 48x22
        self.up2 = nn.ConvTranspose2d(base,     out_ch, 4, stride=2, padding=1)  # 48x22 -> 96x44
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.act(self.from_latent(z))
        for block in self.blocks:
            h = block(h)
        h = self.act(self.up1(h))
        h = self.up2(h)                   # (B, out_ch, 96, 44) — no final activation (regression)
        return h[:, :, 1:-1, :]           # unpad: 96 -> 94


class FNOAutoencoder(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 32, latent_ch: int = 8,
                 modes1: int = 12, modes2: int = 6, n_blocks: int = 2):
        super().__init__()
        self.encoder = FNOEncoder(in_ch, base, latent_ch, modes1, modes2, n_blocks)
        self.decoder = FNODecoder(in_ch, base, latent_ch, modes1, modes2, n_blocks)
        self.latent_ch = latent_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
