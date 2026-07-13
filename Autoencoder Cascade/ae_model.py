import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ResBlockAE(nn.Module):
    """Residual block without timestep conditioning (autoencoder variant)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class RepaintAutoencoder(nn.Module):
    """
    Autoencoder adapted from Repaint UNet shape/layout.

    Input channels:
      2 channels masked velocity + 1 mask channel = 3
    Output channels:
      2 channels reconstructed velocity field

    Spatial shape expected: (94, 44), internally padded to (96, 48).
    """

    _PAD = (2, 2, 1, 1)

    def __init__(self, in_ch: int = 3, out_ch: int = 2, base_ch: int = 64):
        super().__init__()
        c = base_ch

        self.enc0 = ResBlockAE(in_ch, c)
        self.enc1 = ResBlockAE(c, c * 2)
        self.enc2 = ResBlockAE(c * 2, c * 4)
        self.enc3 = ResBlockAE(c * 4, c * 8)

        self.mid = ResBlockAE(c * 8, c * 8)

        self.dec3 = ResBlockAE(c * 8 + c * 8, c * 4)
        self.dec2 = ResBlockAE(c * 4 + c * 4, c * 2)
        self.dec1 = ResBlockAE(c * 2 + c * 2, c)
        self.dec0 = ResBlockAE(c + c, c)

        self.out_conv = nn.Conv2d(c, out_ch, 1)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self._PAD)

        e0 = self.enc0(x)
        e1 = self.enc1(self.down(e0))
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))

        h = self.mid(self.down(e3))

        h = self.up(h)
        h = self.dec3(torch.cat([h, e3], dim=1))

        h = self.up(h)
        h = self.dec2(torch.cat([h, e2], dim=1))

        h = self.up(h)
        h = self.dec1(torch.cat([h, e1], dim=1))

        h = self.up(h)
        h = self.dec0(torch.cat([h, e0], dim=1))

        h = self.out_conv(h)
        return h[:, :, 1:-1, 2:-2]
