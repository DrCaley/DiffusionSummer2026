"""
DDPM diffusion utilities for the white-noise experiment.

Noise schedule : linear (beta_1=1e-4 → beta_T=0.02)
Noise type     : white — i.i.d. Gaussian, flat power spectrum (S(f) ∝ f^0).
                 Standard torch.randn; no spatial correlation.
Loss           : eps-MSE  +  CURL_DIV_WEIGHT * curl_div_loss
"""

import torch
import torch.nn.functional as F

from loss_functions import curl_div_loss

# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
CURL_DIV_WEIGHT = 0.002
NOISE_TYPE      = "white"


# ---------------------------------------------------------------------------
# DDPM — linear schedule + white noise
# ---------------------------------------------------------------------------

class DDPM:
    """
    DDPM with a linear beta schedule and white (flat-spectrum) noise.

    Beta schedule : linear, beta_1=1e-4 → beta_T=0.02
    Noise type    : white — torch.randn (equal power at all frequencies)
    Loss          : eps-MSE + CURL_DIV_WEIGHT * curl_div_loss
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
    # Noise generation
    # ------------------------------------------------------------------

    def _make_noise(self, x0: torch.Tensor) -> torch.Tensor:
        """White noise: i.i.d. Gaussian, flat power spectrum."""
        return torch.randn_like(x0) * self.noise_std

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) = N(sqrt(ᾱ_t)*x0, (1-ᾱ_t)*I)."""
        if noise is None:
            noise = self._make_noise(x0)
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
        return mean + var.sqrt() * torch.randn_like(xt) * self.noise_std

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
        return (
            alpha_eff.sqrt() * x_prev
            + (1.0 - alpha_eff).sqrt() * torch.randn_like(x_prev) * self.noise_std
        )
