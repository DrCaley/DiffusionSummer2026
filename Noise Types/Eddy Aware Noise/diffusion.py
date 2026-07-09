"""
DDPM diffusion utilities for the eddy-aware noise experiment.

Noise schedule : linear (beta_1=1e-4 → beta_T=0.02)
Noise type     : eddy-aware — spatially non-uniform white Gaussian noise.
                 Each grid cell is assigned a noise amplitude σ(x,y) equal to
                 the per-cell velocity standard deviation computed from the
                 training split.  High-energy eddy regions receive larger σ;
                 quiescent background regions receive smaller σ.

                 The sigma_map is computed once at startup from the training
                 data (passed as a pre-computed (H, W) tensor), normalised so
                 that the mean ocean-cell σ equals noise_std (the global ocean
                 std, same as used by white/red/annealed models).

                 At every noise injection:
                     ε(x,y) ~ N(0, σ(x,y)²)
                 so cells are still locally Gaussian and uncorrelated (→ merge
                 consistency for RePaint is preserved), but the amplitude
                 varies with physical energy.

Loss           : eps-MSE  +  CURL_DIV_WEIGHT * curl_div_loss
"""

import torch
import torch.nn.functional as F

from loss_functions import curl_div_loss

# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
CURL_DIV_WEIGHT = 0.002
NOISE_TYPE      = "eddy_aware"


# ---------------------------------------------------------------------------
# DDPM — linear schedule + eddy-aware spatially-varying noise
# ---------------------------------------------------------------------------

class DDPM:
    """
    DDPM with a linear beta schedule and eddy-aware noise.

    The forward process uses spatially non-uniform noise:
        ε(x,y) ~ N(0, σ(x,y)²)
    where σ(x,y) is proportional to the per-cell velocity std across the
    training set.  This means eddy-rich cells are corrupted faster (higher
    σ), giving the model an eddy-focused curriculum during training.

    The noise is still spatially uncorrelated (i.i.d. across cells), so
    RePaint merges are self-consistent without resampling.

    Args:
        sigma_map : (H, W) float32 tensor — per-cell noise std, pre-normalised
                    so mean(ocean cells) == noise_std.  Land cells should be 0.
    """

    def __init__(self, T: int = 1000, device: str = "cpu",
                 noise_std: float = 1.0,
                 curl_div_weight: float = CURL_DIV_WEIGHT,
                 sigma_map: torch.Tensor | None = None):
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

        # sigma_map: (1, 1, H, W) for broadcasting over (B, C, H, W)
        if sigma_map is not None:
            self.sigma_map = sigma_map.to(device)[None, None]  # (1,1,H,W)
        else:
            # Fallback to flat white noise if no map provided
            self.sigma_map = None

    # ------------------------------------------------------------------
    # Noise generation
    # ------------------------------------------------------------------

    def _make_noise(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Eddy-aware noise: spatially non-uniform white Gaussian.

        Each cell is drawn i.i.d. from N(0, σ(x,y)²).
        Land cells (σ=0) produce zero noise automatically.
        """
        white = torch.randn_like(x0)
        if self.sigma_map is not None:
            return white * self.sigma_map
        return white * self.noise_std

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε."""
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
        # Reverse-step stochasticity uses eddy-aware sigma_map
        noise_step = torch.randn_like(xt)
        if self.sigma_map is not None:
            noise_step = noise_step * self.sigma_map
        else:
            noise_step = noise_step * self.noise_std
        return mean + var.sqrt() * noise_step

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
        noise_step = torch.randn_like(x_prev)
        if self.sigma_map is not None:
            noise_step = noise_step * self.sigma_map
        else:
            noise_step = noise_step * self.noise_std
        return alpha_eff.sqrt() * x_prev + (1.0 - alpha_eff).sqrt() * noise_step
