"""
vae_model.py  —  Variational Autoencoder for ocean current fields.

Architecture:
  Encoder: (B, 2, 94, 44) → pad to (96, 48) → 2× stride-2 conv → (B, 2*C_lat, 24, 12)
           → split into mu, logvar each (B, C_lat, 24, 12)
  Decoder: (B, C_lat, 24, 12) → 2× stride-2 ConvTranspose → (B, 2, 96, 48) → crop

C_lat=4 gives ~8× spatial compression (8272 → 1152 elements).
"""
import torch
import torch.nn as nn


def _pad_to(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Pad x from (B, C, h, w) to (B, C, H, W) — pad on right/bottom."""
    ph = H - x.shape[2]
    pw = W - x.shape[3]
    return nn.functional.pad(x, (0, pw, 0, ph))


def _crop_to(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return x[:, :, :H, :W]


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 2, base_ch: int = 32, c_lat: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            ResBlock(base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),   # /2
            ResBlock(base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),  # /4
            ResBlock(base_ch * 4),
            nn.GroupNorm(min(8, base_ch * 4), base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, c_lat * 2, 3, padding=1),            # mu + logvar
        )

    def forward(self, x):
        h   = self.net(x)
        mu, logvar = h.chunk(2, dim=1)
        logvar = logvar.clamp(-30, 20)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch: int = 2, base_ch: int = 32, c_lat: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_lat, base_ch * 4, 3, padding=1),
            ResBlock(base_ch * 4),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),  # ×2
            ResBlock(base_ch * 2),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),       # ×4
            ResBlock(base_ch),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class OceanVAE(nn.Module):
    """
    Variational Autoencoder for (B, 2, 94, 44) ocean current fields.
    Internally pads to (96, 48) for 4× spatial downsampling compatibility.
    Latent shape: (B, c_lat, 24, 12).
    """
    PAD_H = 96
    PAD_W = 48

    def __init__(self, c_lat: int = 4, base_ch: int = 32):
        super().__init__()
        self.c_lat   = c_lat
        self.base_ch = base_ch
        self.encoder = Encoder(in_ch=2,    base_ch=base_ch, c_lat=c_lat)
        self.decoder = Decoder(out_ch=2,   base_ch=base_ch, c_lat=c_lat)

    @property
    def latent_shape(self):
        return (self.c_lat, self.PAD_H // 4, self.PAD_W // 4)  # (c_lat, 24, 12)

    def encode(self, x: torch.Tensor):
        """x: (B, 2, 94, 44) → mu, logvar each (B, c_lat, 24, 12)"""
        xp = _pad_to(x, self.PAD_H, self.PAD_W)
        return self.encoder(xp)

    def decode(self, z: torch.Tensor, orig_H: int = 94, orig_W: int = 44):
        """z: (B, c_lat, 24, 12) → (B, 2, 94, 44)"""
        xp = self.decoder(z)
        return _crop_to(xp, orig_H, orig_W)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decode(z, orig_H=x.shape[2], orig_W=x.shape[3])
        return recon, mu, logvar

    def sample_prior(self, n: int, device) -> torch.Tensor:
        """Sample n latent codes from N(0,I), return (n, c_lat, 24, 12)."""
        C, H, W = self.latent_shape
        return torch.randn(n, C, H, W, device=device)
