import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timestep (B,) to sinusoidal embedding (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]  # (B, half)
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


def _num_groups(channels: int) -> int:
    """Pick the largest divisor of channels that is <= 32."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with time-step conditioning via addition."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1   = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_fc = nn.Linear(time_dim, out_ch)
        self.norm2   = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act     = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_fc(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    2D UNet for DDPM noise prediction on 2-channel ocean current fields.

    Input spatial size: (H, W) = (94, 44).
    Internally padded to (96, 48) for clean factor-of-2 downsampling:
        96 → 48 → 24 → 12 → 6 → 3  (bottleneck)
        48 → 24 → 12 → 6            (width)

    Padding applied: left=2, right=2, top=1, bottom=1 → 94→96 height, 44→48 width.
    """

    # (left, right, top, bottom) — F.pad convention is last-dim first
    _PAD  = (2, 2, 1, 1)   # → 44+4=48 wide, 94+2=96 tall
    _UPAD = (2, 2, 1, 1)   # same amounts to strip when unpadding

    def __init__(self, in_ch: int = 2, base_ch: int = 64, time_dim: int = 256,
                 out_ch: int | None = None):
        super().__init__()
        self.time_dim = time_dim
        # Output channels default to the input count (eps / x0 in (u,v) space).
        # Pass out_ch=1 to emit a single scalar (e.g. a stream function).
        out_channels = in_ch if out_ch is None else out_ch
        c = base_ch

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.enc0 = ResBlock(in_ch,  c,    time_dim)   # 96×48
        self.enc1 = ResBlock(c,      c*2,  time_dim)   # 48×24
        self.enc2 = ResBlock(c*2,    c*4,  time_dim)   # 24×12
        self.enc3 = ResBlock(c*4,    c*8,  time_dim)   # 12×6

        # Bottleneck
        self.mid  = ResBlock(c*8,    c*8,  time_dim)   # 6×3

        # Decoder  (skip connections double the input channels)
        self.dec3 = ResBlock(c*8+c*8, c*4, time_dim)   # 12×6
        self.dec2 = ResBlock(c*4+c*4, c*2, time_dim)   # 24×12
        self.dec1 = ResBlock(c*2+c*2, c,   time_dim)   # 48×24
        self.dec0 = ResBlock(c  +c,   c,   time_dim)   # 96×48

        self.out_conv = nn.Conv2d(c, out_channels, 1)

        self.down = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 94, 44) noisy field
            t: (B,) integer timesteps
        Returns:
            predicted noise: (B, 2, 94, 44)
        """
        # Pad to 96×48
        x = F.pad(x, self._PAD)

        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        e0 = self.enc0(x,              t_emb)   # 96×48
        e1 = self.enc1(self.down(e0),  t_emb)   # 48×24
        e2 = self.enc2(self.down(e1),  t_emb)   # 24×12
        e3 = self.enc3(self.down(e2),  t_emb)   # 12×6

        # Bottleneck
        h = self.mid(self.down(e3), t_emb)      # 6×3

        # Decoder
        h = self.up(h)                                           # 12×6
        h = self.dec3(torch.cat([h, e3], dim=1), t_emb)

        h = self.up(h)                                           # 24×12
        h = self.dec2(torch.cat([h, e2], dim=1), t_emb)

        h = self.up(h)                                           # 48×24
        h = self.dec1(torch.cat([h, e1], dim=1), t_emb)

        h = self.up(h)                                           # 96×48
        h = self.dec0(torch.cat([h, e0], dim=1), t_emb)

        h = self.out_conv(h)

        # Unpad back to 94×44
        # _PAD = (left=2, right=2, top=1, bottom=1)
        # so strip: height [1:-1], width [2:-2]
        h = h[:, :, 1:-1, 2:-2]
        return h


# ---------------------------------------------------------------------------
# Stream-function UNet — divergence-free vector-field predictor
# ---------------------------------------------------------------------------

class StreamFunctionUNet(nn.Module):
    """
    Divergence-free vector-field predictor for incompressible ocean currents.

    A UNet backbone maps the noisy field x_t (in_ch channels) to a single
    scalar stream function ψ (1 channel).  The discrete curl of ψ yields a
    2-channel (u, v) field that is divergence-free *by construction*:

        u =  ∂ψ/∂y  =  ∂ψ/∂W   (W-direction central difference)
        v = -∂ψ/∂x  = -∂ψ/∂H   (H-direction central difference)

    The central-difference kernels match divfree_projection._kW / _kH, so the
    output has exactly zero central-difference divergence in the interior
    (boundary cells carry the usual one-sided error).  No Leray projection is
    ever required.

    This parameterises the clean field x̂₀ directly (x0-prediction), making the
    network a *calibrated* denoiser whose output is guaranteed incompressible —
    ideal for coherent / divergence-free eddy structure.
    """

    def __init__(self, in_ch: int = 2, base_ch: int = 64, time_dim: int = 256,
                 cond_ch: int = 0):
        super().__init__()
        self.in_ch   = in_ch
        self.cond_ch = cond_ch
        # The backbone consumes the noisy field (in_ch) plus any conditioning
        # channels (cond_ch).  It always emits a single scalar stream function.
        self.backbone = UNet(in_ch=in_ch + cond_ch, base_ch=base_ch,
                             time_dim=time_dim, out_ch=1)

        # Central-difference kernels (identical to divfree_projection._kH/_kW).
        kH = torch.tensor(
            [[[[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]]]]) / 2.0   # ∂/∂H = ∂/∂x
        kW = torch.tensor(
            [[[[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]]]]) / 2.0   # ∂/∂W = ∂/∂y
        self.register_buffer("_kH", kH)
        self.register_buffer("_kW", kW)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    (B, in_ch, 94, 44) noisy field x_t
            t:    (B,) integer timesteps
            cond: (B, cond_ch, 94, 44) optional conditioning channels, stacked
                  onto x before the backbone.  Must be provided iff the model
                  was built with cond_ch > 0.
        Returns:
            (B, 2, 94, 44) divergence-free predicted clean field x̂₀
        """
        if self.cond_ch > 0:
            if cond is None:
                raise ValueError(
                    f"model built with cond_ch={self.cond_ch} but cond is None")
            inp = torch.cat([x, cond], dim=1)           # (B, in_ch+cond_ch, H, W)
        else:
            if cond is not None:
                raise ValueError("model built with cond_ch=0 but cond was given")
            inp = x
        psi = self.backbone(inp, t)                     # (B, 1, H, W)
        u   =  F.conv2d(psi, self._kW, padding=1)       # ∂ψ/∂y
        v   = -F.conv2d(psi, self._kH, padding=1)       # -∂ψ/∂x
        return torch.cat([u, v], dim=1)                 # (B, 2, H, W)
