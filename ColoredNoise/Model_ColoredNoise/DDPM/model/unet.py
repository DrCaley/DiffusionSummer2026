"""
unet.py – UNet for 2-D ocean-current DDPM (epsilon-prediction).

Architecture
------------
Input  : (B, 2, 94, 44) – padded internally to (B, 2, 96, 48).
Output : (B, 2, 94, 44) – predicted noise, same shape as input.

Encoder (3 levels):
  Level 0 [96×48]:  init_conv → 2×ResBlock(128,128) → stride-2 Conv
  Level 1 [48×24]:  2×ResBlock(256,256) → stride-2 Conv
  Level 2 [24×12]:  2×ResBlock(256,256)  [feeds bottleneck, no downsample]

Bottleneck [24×12]:
  ResBlock → SelfAttention → ResBlock

Decoder (3 levels, skip connections from encoder):
  Level 2 [24×12]:  cat(h, skip2) → 2×ResBlock → Upsample
  Level 1 [48×24]:  cat(h, skip1) → 2×ResBlock → Upsample
  Level 0 [96×48]:  cat(h, skip0) → 2×ResBlock

Output head:
  GroupNorm → SiLU → Conv1×1(128→2) → crop to (94, 44)

Time embedding:
  sinusoidal(base_ch=128) → Linear(128→512) → SiLU → Linear(512→512)

Parameter count ≈ 14.9 M.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding.

    Parameters
    ----------
    t   : LongTensor (B,)
    dim : int  (must be even)

    Returns
    -------
    FloatTensor (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]            # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU activations, and time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        g_in  = min(32, in_ch)
        g_out = min(32, out_ch)

        self.norm1  = nn.GroupNorm(g_in, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(time_dim, out_ch)
        self.norm2  = nn.GroupNorm(g_out, out_ch)
        self.drop   = nn.Dropout(dropout)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention applied over spatial tokens."""

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        assert ch % num_heads == 0
        self.norm      = nn.GroupNorm(min(32, ch), ch)
        self.qkv       = nn.Conv2d(ch, ch * 3, 1)
        self.proj_out  = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads
        self.head_dim  = ch // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)                              # (B, heads, head_dim, HW)
        scale = self.head_dim ** -0.5
        attn  = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale  # (B, heads, HW, HW)
        attn  = attn.softmax(dim=-1)
        out   = torch.einsum("bhnm,bhdm->bhdn", attn, v)        # (B, heads, head_dim, HW)
        out   = out.reshape(B, C, H, W)
        return x + self.proj_out(out)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Epsilon-predicting UNet for the colored-noise DDPM.

    Parameters
    ----------
    in_channels : int    Input/output channels (2 for (u, v) field).
    base_ch     : int    Base channel count (Level 0).
    ch_mults    : tuple  Channel multipliers per encoder level.
    time_dim    : int    Time embedding MLP output dimension.
    dropout     : float  Dropout probability inside ResBlocks.
    """

    def __init__(
        self,
        in_channels: int  = 2,
        base_ch    : int  = 128,
        ch_mults   : tuple = (1, 2, 2),
        time_dim   : int  = 512,
        dropout    : float = 0.1,
    ):
        super().__init__()

        C0, C1, C2 = [base_ch * m for m in ch_mults]   # 128, 256, 256

        # --- Time embedding ---
        self.time_embed = nn.Sequential(
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- Initial projection ---
        self.init_conv = nn.Conv2d(in_channels, C0, 3, padding=1)

        # --- Encoder ---
        # Level 0 (96×48)
        self.enc0_0 = ResBlock(C0, C0, time_dim, dropout)
        self.enc0_1 = ResBlock(C0, C0, time_dim, dropout)
        self.down0  = nn.Conv2d(C0, C0, 3, stride=2, padding=1)   # → 48×24

        # Level 1 (48×24)
        self.enc1_0 = ResBlock(C0, C1, time_dim, dropout)
        self.enc1_1 = ResBlock(C1, C1, time_dim, dropout)
        self.down1  = nn.Conv2d(C1, C1, 3, stride=2, padding=1)   # → 24×12

        # Level 2 (24×12)
        self.enc2_0 = ResBlock(C1, C2, time_dim, dropout)
        self.enc2_1 = ResBlock(C2, C2, time_dim, dropout)

        # --- Bottleneck (24×12) ---
        self.mid1     = ResBlock(C2, C2, time_dim, dropout)
        self.mid_attn = SelfAttention(C2)
        self.mid2     = ResBlock(C2, C2, time_dim, dropout)

        # --- Decoder ---
        # Level 2
        self.dec2_0 = ResBlock(C2 + C2, C2, time_dim, dropout)
        self.dec2_1 = ResBlock(C2,      C2, time_dim, dropout)
        self.up2    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C2, C2, 3, padding=1),
        )

        # Level 1
        self.dec1_0 = ResBlock(C2 + C1, C1, time_dim, dropout)
        self.dec1_1 = ResBlock(C1,      C1, time_dim, dropout)
        self.up1    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C1, C0, 3, padding=1),
        )

        # Level 0
        self.dec0_0 = ResBlock(C0 + C0, C0, time_dim, dropout)
        self.dec0_1 = ResBlock(C0,      C0, time_dim, dropout)

        # --- Output head ---
        self.out_norm = nn.GroupNorm(min(32, C0), C0)
        self.out_conv = nn.Conv2d(C0, in_channels, 1)

        self._base_ch = base_ch

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor (B, 2, 94, 44)
        t : LongTensor  (B,)

        Returns
        -------
        FloatTensor (B, 2, 94, 44) – predicted noise
        """
        # Pad: H 94→96 (+1 top, +1 bot), W 44→48 (+2 left, +2 right)
        x = F.pad(x, (2, 2, 1, 1))

        # Time embedding
        t_emb = sinusoidal_embedding(t, self._base_ch)   # (B, base_ch)
        t_emb = self.time_embed(t_emb)                    # (B, time_dim)

        # Encoder
        h = self.init_conv(x)          # (B, C0, 96, 48)

        h      = self.enc0_0(h, t_emb)
        h      = self.enc0_1(h, t_emb)
        skip0  = h                     # (B, C0, 96, 48)
        h      = self.down0(h)         # (B, C0, 48, 24)

        h      = self.enc1_0(h, t_emb)
        h      = self.enc1_1(h, t_emb)
        skip1  = h                     # (B, C1, 48, 24)
        h      = self.down1(h)         # (B, C1, 24, 12)

        h      = self.enc2_0(h, t_emb)
        h      = self.enc2_1(h, t_emb)
        skip2  = h                     # (B, C2, 24, 12)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)        # (B, C2, 24, 12)

        # Decoder
        h = torch.cat([h, skip2], dim=1)
        h = self.dec2_0(h, t_emb)
        h = self.dec2_1(h, t_emb)
        h = self.up2(h)                # (B, C2, 48, 24)

        h = torch.cat([h, skip1], dim=1)
        h = self.dec1_0(h, t_emb)
        h = self.dec1_1(h, t_emb)
        h = self.up1(h)                # (B, C0, 96, 48)

        h = torch.cat([h, skip0], dim=1)
        h = self.dec0_0(h, t_emb)
        h = self.dec0_1(h, t_emb)

        # Output
        h = self.out_conv(F.silu(self.out_norm(h)))   # (B, 2, 96, 48)

        # Crop back to (B, 2, 94, 44)
        h = h[:, :, 1:-1, 2:-2]

        return h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
