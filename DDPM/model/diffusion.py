import math
import torch
import torch.nn.functional as F


class DDPM:
    """
    Denoising Diffusion Probabilistic Model utilities.

    Handles the cosine noise schedule, forward process q(x_t | x_0),
    training loss, and a single reverse step p(x_{t-1} | x_t).
    """

    def __init__(self, T: int = 1000, beta_schedule: str = "cosine", device: str = "cpu"):
        self.T      = T
        self.device = device

        betas = self._cosine_betas(T) if beta_schedule == "cosine" else \
                torch.linspace(1e-4, 0.02, T)

        self.betas    = betas.to(device)
        alphas        = 1.0 - self.betas
        self.alphas   = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)           # ᾱ_t
        # Prepend 1.0 so alpha_bar_prev[t] = ᾱ_{t-1} (alpha_bar_prev[0] = 1)
        self.alpha_bar_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bar[:-1]]
        )
        self.sqrt_ab      = self.alpha_bar.sqrt()
        self.sqrt_one_mab = (1.0 - self.alpha_bar).sqrt()

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _cosine_betas(self, T: int, s: float = 0.008) -> torch.Tensor:
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = 1.0 - ab[1:] / ab[:-1]
        return betas.clamp(0, 0.999)

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
            noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_ab[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_mab[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_mab * noise, noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model:     torch.nn.Module,
        x0:        torch.Tensor,
        land_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simple epsilon-prediction MSE loss, computed only on ocean pixels.

        Args:
            model:     UNet that predicts noise
            x0:        (B, 2, H, W) clean fields
            land_mask: (H, W) bool, True = land (excluded from loss)
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)

        # Ocean mask broadcast to (1, 1, H, W)
        ocean = (~land_mask).float()[None, None]
        loss = F.mse_loss(pred_noise * ocean, noise * ocean)
        return loss

    # ------------------------------------------------------------------
    # Single reverse step  p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(
        self,
        model: torch.nn.Module,
        xt:    torch.Tensor,
        t_int: int,
    ) -> torch.Tensor:
        """
        One DDPM reverse step.

        Args:
            model: trained UNet
            xt:    (B, 2, H, W) current noisy field
            t_int: integer timestep (same for whole batch)
        Returns:
            x_{t-1}: (B, 2, H, W)
        """
        B = xt.shape[0]
        t = torch.full((B,), t_int, device=self.device, dtype=torch.long)

        pred_noise = model(xt, t)

        ab      = self.alpha_bar[t_int]
        ab_prev = self.alpha_bar_prev[t_int]
        beta    = self.betas[t_int]
        alpha   = self.alphas[t_int]

        # Predicted x0 (clipped)
        x0_pred = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        if t_int == 0:
            return x0_pred

        # DDPM posterior mean
        coef1 = ab_prev.sqrt() * beta / (1.0 - ab)
        coef2 = alpha.sqrt() * (1.0 - ab_prev) / (1.0 - ab)
        mean  = coef1 * x0_pred + coef2 * xt

        # Posterior variance
        var = (1.0 - ab_prev) / (1.0 - ab) * beta
        return mean + var.sqrt() * torch.randn_like(xt)

    # ------------------------------------------------------------------
    # One forward step  q(x_t | x_{t-1})  — used by RePaint resampling
    # ------------------------------------------------------------------

    def q_sample_from_prev(self, x_prev: torch.Tensor, t_int: int) -> torch.Tensor:
        """
        Add one step of noise: x_t ~ q(x_t | x_{t-1}).
        x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * eps
        """
        alpha = self.alphas[t_int]
        return alpha.sqrt() * x_prev + (1.0 - alpha).sqrt() * torch.randn_like(x_prev)
