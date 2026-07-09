"""
DDPM diffusion utilities for the annealed-noise experiment.

Noise schedule : linear (beta_1=1e-4 → beta_T=0.02)
Noise type     : annealed — spectral exponent α(t) = 2·t/T
                 At t=T  → red noise (α=2, smooth large-scale)
                 At t=T/2→ pink noise (α=1)
                 At t=0  → white noise (α=0, no spatial correlation)

Rationale: early diffusion steps destroy large-scale structure (red),
           later steps destroy fine detail (white), mirroring the
           multi-scale energy spectrum of ocean turbulence.

ALL noise injections (forward, reverse posterior, RePaint resampling)
use the t-appropriate noise color consistently.

Loss : eps-MSE  +  CURL_DIV_WEIGHT * curl_div_loss
"""

import torch
import torch.nn.functional as F

from loss_functions import curl_div_loss

# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
CURL_DIV_WEIGHT = 0.002
NOISE_TYPE      = "annealed"


# ---------------------------------------------------------------------------
# DDPM — linear schedule + annealed colored noise
# ---------------------------------------------------------------------------

class DDPM:
    """
    DDPM with a linear beta schedule and annealed noise color.

    The spectral exponent α varies continuously with timestep t:
        α(t) = 2 · t / T      (0 = white, 1 = pink, 2 = red)

    This means:
      - Forward process (large t):  red/smooth noise destroys large-scale structure
      - Forward process (small t):  white noise destroys fine-scale detail
      - Reverse process mirrors this: same α(t) used at every injection
    """

    def __init__(self, T: int = 1000, device: str = "cpu",
                 noise_std: float = 1.0,
                 curl_div_weight: float = CURL_DIV_WEIGHT):
        self.T               = T
        self.device          = device
        self.noise_std       = noise_std
        self.curl_div_weight = curl_div_weight

        betas             = torch.linspace(1e-4, 0.02, T)
        self.betas        = betas.to(device)
        alphas            = 1.0 - self.betas
        self.alphas       = alphas
        self.alpha_bar    = torch.cumprod(alphas, dim=0)
        self.alpha_bar_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bar[:-1]]
        )
        self.sqrt_ab      = self.alpha_bar.sqrt()
        self.sqrt_one_mab = (1.0 - self.alpha_bar).sqrt()

    # ------------------------------------------------------------------
    # Noise generation — t-dependent spectral exponent
    # ------------------------------------------------------------------

    def _make_noise(self, x: torch.Tensor, t_int: int | None = None) -> torch.Tensor:
        """
        Annealed colored noise: spectral exponent α(t) = 2·t/T.

        At t=T-1 → α≈2 (red/Brownian, smooth)
        At t=T/2 → α≈1 (pink, 1/f)
        At t=0   → α=0 (white, no correlation)

        Generates white noise, applies 2-D spectral filter with weight
        ∝ (|f|² + ε)^{-α/2}, normalises to noise_std.
        DC component zeroed to prevent mean drift.

        Args:
            x:     tensor whose shape (B,C,H,W) and device are used
            t_int: integer timestep in [0, T-1]; None defaults to T-1 (reddest)
        """
        if t_int is None:
            t_int = self.T - 1

        alpha = 2.0 * t_int / (self.T - 1)   # 0.0 at t=0, 2.0 at t=T-1

        B, C, H, W = x.shape
        device = x.device
        white  = torch.randn(B, C, H, W, device=device)

        if alpha == 0.0:
            # Pure white noise — no filtering needed
            std = white.std() + 1e-8
            return white / std * self.noise_std

        # Build 2-D radial frequency grid
        fy = torch.fft.fftfreq(H, device=device)       # (H,)
        fx = torch.fft.fftfreq(W, device=device)       # (W,)
        FY, FX   = torch.meshgrid(fy, fx, indexing="ij")   # (H, W)
        f_sq     = FY ** 2 + FX ** 2                   # (H, W)

        # Spectral filter: amplitude ∝ (|f|² + ε)^{-α/2}
        eps  = 1e-10
        filt = (f_sq + eps) ** (-alpha / 2.0)
        filt[f_sq == 0] = 0.0           # zero DC (no mean offset)
        filt = filt[None, None]         # (1, 1, H, W) for broadcasting

        noise_fft = torch.fft.fft2(white)
        noise_fft = noise_fft * filt
        colored   = torch.fft.ifft2(noise_fft).real   # (B, C, H, W)

        std = colored.std() + 1e-8
        return colored / std * self.noise_std

    def _make_noise_batch(self, x0: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
        """
        Generate a batch of noise samples where each sample uses the α
        appropriate for its own timestep t_batch[i].

        Args:
            x0:      (B, C, H, W)
            t_batch: (B,) integer timesteps

        Returns:
            noise:   (B, C, H, W) with per-sample colored noise
        """
        pieces = []
        for i in range(x0.shape[0]):
            pieces.append(self._make_noise(x0[i:i+1], t_batch[i].item()))
        return torch.cat(pieces, dim=0)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) using t-appropriate noise color."""
        if noise is None:
            noise = self._make_noise_batch(x0, t)
        sqrt_ab  = self.sqrt_ab[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_mab[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_mab * noise, noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model:     torch.nn.Module,
        x0:        torch.Tensor,
        land_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)

        ocean    = (~land_mask).float()[None, None]
        eps_loss = F.mse_loss(pred_noise * ocean, noise * ocean)

        if self.curl_div_weight == 0.0:
            return eps_loss

        ab     = self.alpha_bar[t][:, None, None, None]
        x0_hat = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        cd_loss = curl_div_loss(x0_hat, x0, ocean)
        return eps_loss + self.curl_div_weight * cd_loss

    # ------------------------------------------------------------------
    # Single reverse step  p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(
        self,
        model:      torch.nn.Module,
        xt:         torch.Tensor,
        t_int:      int,
        t_prev_int: int | None = None,
    ) -> torch.Tensor:
        if t_prev_int is None:
            t_prev_int = max(t_int - 1, 0)

        B = xt.shape[0]
        t = torch.full((B,), t_int, device=self.device, dtype=torch.long)

        pred_noise = model(xt, t)

        ab      = self.alpha_bar[t_int]
        ab_prev = (
            self.alpha_bar[t_prev_int]
            if t_prev_int > 0
            else torch.tensor(1.0, device=self.device)
        )

        alpha_eff = ab / ab_prev
        beta_eff  = 1.0 - alpha_eff

        x0_pred = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1.5, 1.5)

        if t_int == 0:
            return x0_pred

        coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
        coef2 = alpha_eff.sqrt() * (1.0 - ab_prev) / (1.0 - ab)
        mean  = coef1 * x0_pred + coef2 * xt

        var = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
        # Inject noise at the color appropriate for t_prev_int
        return mean + var.sqrt() * self._make_noise(xt, t_prev_int)

    # ------------------------------------------------------------------
    # One forward step  q(x_t | x_{t-1})  — used by RePaint resampling
    # ------------------------------------------------------------------

    def q_sample_from_prev(
        self, x_prev: torch.Tensor, t_int: int, t_prev_int: int = -1
    ) -> torch.Tensor:
        if t_prev_int < 0:
            t_prev_int = max(t_int - 1, 0)
        ab      = self.alpha_bar[t_int]
        ab_prev = (
            self.alpha_bar[t_prev_int]
            if t_prev_int > 0
            else torch.tensor(1.0, device=self.device)
        )
        alpha_eff = ab / ab_prev
        # Inject noise at color appropriate for t_int
        return (
            alpha_eff.sqrt() * x_prev
            + (1.0 - alpha_eff).sqrt() * self._make_noise(x_prev, t_int)
        )
