"""
latent_unet.py  —  Lightweight 2-level UNet for latent DDPM denoising.

Designed for latent tensors of shape (B, C_lat, H_lat, W_lat).
Uses 2 downsampling stages so only needs H, W divisible by 4.
  (24, 12): 24/4=6 ✓  12/4=3 ✓  — works without padding.

No hardcoded spatial assumptions — general purpose latent denoiser.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args  = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1  = nn.GroupNorm(min(8, in_ch),  in_ch)
        self.norm2  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(time_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act    = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class LatentUNet(nn.Module):
    """
    2-level UNet for latent DDPM.

    Encoder: in_ch → c → 2c → 4c  (2× MaxPool downsamplings)
    Bottleneck: 4c → 4c
    Decoder: (4c+4c) → 2c → (2c+2c) → c → out_ch
    """

    def __init__(self, in_ch: int = 4, base_ch: int = 64, time_dim: int = 256):
        super().__init__()
        c = base_ch
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder (2 downsampling ops → needs H,W divisible by 4)
        self.enc0 = ResBlock(in_ch, c,    time_dim)   # H,   W
        self.enc1 = ResBlock(c,     c*2,  time_dim)   # H/2, W/2
        self.enc2 = ResBlock(c*2,   c*4,  time_dim)   # H/4, W/4

        # Bottleneck (no extra downsampling)
        self.mid  = ResBlock(c*4,   c*4,  time_dim)   # H/4, W/4

        # Decoder (2 upsampling ops, skip connections from enc1 and enc0)
        self.dec2 = ResBlock(c*4+c*2, c*2, time_dim)  # H/2, W/2  (up from H/4 + e1)
        self.dec1 = ResBlock(c*2+c,   c,   time_dim)  # H,   W    (up from H/2 + e0)

        self.out  = nn.Conv2d(c, in_ch, 1)
        self.down = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # 2 pooling ops → needs H,W divisible by 4
        e0 = self.enc0(x,             t_emb)   # (B, c,  H,   W  )
        e1 = self.enc1(self.down(e0), t_emb)   # (B, 2c, H/2, W/2)
        e2 = self.enc2(self.down(e1), t_emb)   # (B, 4c, H/4, W/4)

        h  = self.mid(e2, t_emb)               # (B, 4c, H/4, W/4) — NO extra down

        h = self.up(h)                                          # H/2, W/2
        h = self.dec2(torch.cat([h, e1], dim=1), t_emb)

        h = self.up(h)                                          # H,   W
        h = self.dec1(torch.cat([h, e0], dim=1), t_emb)

        return self.out(h)
