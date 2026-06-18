"""
UNet speed (magnitude) regressor.

A deterministic image-to-image UNet that maps a sparse observed-speed field
(plus path and land masks) to a dense speed field.  Unlike the DDPM UNet this
has NO time-step conditioning — it is a plain regressor.

Input : (B, 3, 94, 44)  channels = [observed_speed, path_mask, land_mask]
Output: (B, 1, 94, 44)  dense speed field (in standardized space; the training
        script un-standardizes with the stored speed mean/std).

The 94×44 grid is padded to 96×48 for clean factor-of-2 down/up-sampling, then
cropped back, mirroring the DDPM UNet's padding scheme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    """Largest divisor of `channels` that is <= 32 (for GroupNorm)."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Residual block: GroupNorm → SiLU → Conv, twice, with a skip."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class MagnitudeUNet(nn.Module):
    """UNet regressor for dense speed reconstruction from sparse observations."""

    # (left, right, top, bottom) → 44+4=48 wide, 94+2=96 tall
    _PAD = (2, 2, 1, 1)

    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        c = base_ch

        # Encoder
        self.enc0 = ResBlock(in_ch, c)       # 96×48
        self.enc1 = ResBlock(c,     c * 2)   # 48×24
        self.enc2 = ResBlock(c * 2, c * 4)   # 24×12
        self.enc3 = ResBlock(c * 4, c * 8)   # 12×6

        # Bottleneck
        self.mid = ResBlock(c * 8, c * 8)    # 6×3

        # Decoder (skip connections double the input channels)
        self.dec3 = ResBlock(c * 8 + c * 8, c * 4)   # 12×6
        self.dec2 = ResBlock(c * 4 + c * 4, c * 2)   # 24×12
        self.dec1 = ResBlock(c * 2 + c * 2, c)       # 48×24
        self.dec0 = ResBlock(c + c,         c)       # 96×48

        self.out_conv = nn.Conv2d(c, 1, 1)

        self.down = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 94, 44) input channels [observed_speed, path_mask, land_mask]
        Returns:
            (B, 1, 94, 44) predicted speed field (standardized space)
        """
        x = F.pad(x, self._PAD)            # → 96×48

        e0 = self.enc0(x)                  # 96×48
        e1 = self.enc1(self.down(e0))      # 48×24
        e2 = self.enc2(self.down(e1))      # 24×12
        e3 = self.enc3(self.down(e2))      # 12×6

        h = self.mid(self.down(e3))        # 6×3

        h = self.up(h)                                       # 12×6
        h = self.dec3(torch.cat([h, e3], dim=1))
        h = self.up(h)                                       # 24×12
        h = self.dec2(torch.cat([h, e2], dim=1))
        h = self.up(h)                                       # 48×24
        h = self.dec1(torch.cat([h, e1], dim=1))
        h = self.up(h)                                       # 96×48
        h = self.dec0(torch.cat([h, e0], dim=1))

        h = self.out_conv(h)               # 96×48, 1 channel

        # Crop back to 94×44 (strip the padding)
        h = h[:, :, 1:-1, 2:-2]
        return h
