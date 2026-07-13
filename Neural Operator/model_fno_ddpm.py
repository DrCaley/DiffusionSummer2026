"""
model_fno_ddpm.py
==================
Noise-conditioned Fourier Neural Operator (FNO-DDPM).

The original model_fno.py FNO2d was trained as a plain autoencoder
(clean field -> clean field) and only turned into a diffusion prior after
the fact via repaint_infer_fno.py's FNOx0Predictor hack. That never saw
noisy inputs during training, which is why it performed far worse than the
UNet (RMSE ~1.4 vs ~0.04-0.07 on this dataset).

FNO2dDDPM instead is trained directly as a proper epsilon-predictor, exactly
like repaint_model.Repaint: forward(x_t, t) -> eps. Same interface, same
(B, 2, H, W) in/out shape, same sinusoidal time embedding — so it drops into
diffusion.py's training_loss / p_sample_step and repaint_infer.py's repaint()
unchanged, just by swapping the model class.

Unlike the UNet, FNO2dDDPM needs no spatial padding: spectral convolutions
via rfft2/irfft2 work at the native (94, 44) resolution directly.
"""

import os
import sys

import torch
import torch.nn as nn

# ── locate repaint_model.py (defines sinusoidal_embedding / _num_groups) ──────
# Same search strategy as repaint_infer_fno.py's _find_diffusion_dir, so this
# works whether run from Neural Operator/ locally or copied next to
# Repaint_vs_DPS on the remote.
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
from repaint_model import sinusoidal_embedding, _num_groups  # noqa: E402


# ---------------------------------------------------------------------------
# Spectral convolution (same truncated-mode complex weight mult as model_fno.py)
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )

    def compl_mul2d(self, inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", inp, cweights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            B, self.weight.shape[1], H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft[..., :m1, :m2] = self.compl_mul2d(
            x_ft[..., :m1, :m2], self.weight[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


# ---------------------------------------------------------------------------
# Time-conditioned spectral block
# ---------------------------------------------------------------------------

class FNOBlock(nn.Module):
    """GroupNorm -> +time embedding -> (spectral conv + pointwise conv) -> residual.

    Mirrors repaint_model.ResBlock's time-conditioning-via-addition, but with
    a spectral convolution (+ a 1x1 pointwise conv, as in the original FNO2d)
    in place of the UNet's 3x3 convs.
    """

    def __init__(self, width: int, modes1: int, modes2: int, time_dim: int):
        super().__init__()
        self.norm    = nn.GroupNorm(_num_groups(width), width)
        self.spec    = SpectralConv2d(width, width, modes1, modes2)
        self.point   = nn.Conv2d(width, width, 1)
        self.time_fc = nn.Linear(time_dim, width)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(x))
        h = h + self.time_fc(self.act(t_emb))[:, :, None, None]
        h = self.act(self.spec(h) + self.point(h))
        return x + h


# ---------------------------------------------------------------------------
# FNO-DDPM
# ---------------------------------------------------------------------------

class FNO2dDDPM(nn.Module):
    """
    Noise-conditioned FNO: FNO(x_t, t) -> eps_pred.

    Drop-in replacement for repaint_model.Repaint — same forward(x, t)
    signature, same (B, 2, H, W) shapes, same time-embedding scheme. No
    padding needed (unlike the UNet's pad to 96x48 for clean 2x
    downsampling); the spectral convs operate at native (94, 44) resolution.
    """

    def __init__(self, in_ch: int = 2, width: int = 64, modes1: int = 16,
                 modes2: int = 16, time_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.fc0    = nn.Linear(in_ch, width)
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2, time_dim) for _ in range(n_layers)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, in_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, H, W) noisy field
            t: (B,) integer timesteps
        Returns:
            predicted noise: (B, 2, H, W)
        """
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = x.permute(0, 2, 3, 1)                   # (B, H, W, C)
        h = self.fc0(h)
        h = h.permute(0, 3, 1, 2).contiguous()       # (B, width, H, W)

        for block in self.blocks:
            h = block(h, t_emb)

        h = h.permute(0, 2, 3, 1)                    # (B, H, W, width)
        h = self.act(self.fc1(h))
        h = self.fc2(h)                              # (B, H, W, in_ch)
        return h.permute(0, 3, 1, 2).contiguous()    # (B, in_ch, H, W)


def get_model(in_ch=2, width=64, modes1=16, modes2=16, time_dim=256,
              n_layers=4, device="cpu"):
    model = FNO2dDDPM(in_ch=in_ch, width=width, modes1=modes1, modes2=modes2,
                      time_dim=time_dim, n_layers=n_layers)
    return model.to(device)
