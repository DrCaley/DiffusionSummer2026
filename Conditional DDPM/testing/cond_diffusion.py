"""
CondDDPM — DDPM utilities for FiLM-conditioned models.

This mirrors the structure of DDPM/model/diffusion.py but all model calls
pass an extra `cond` tensor: model(xt, t, cond).

Noise schedule, q_sample, and reverse sampling are identical to the base
DDPM (cosine schedule by default).  Training uses epsilon-prediction MSE
restricted to ocean pixels (land masked out).
"""

import importlib.util
import math
import os

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Load div_free_noise.py from utils/ via importlib so this file works from any
# working directory.  Searches up to 4 levels above for utils/div_free_noise.py.
# ---------------------------------------------------------------------------
_df_path = None
for _up in range(5):
    _candidate = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        *(['..'] * _up),
        "utils", "div_free_noise.py",
    ))
    if os.path.isfile(_candidate):
        _df_path = _candidate
        break
if _df_path is None:
    raise FileNotFoundError(
        "Cannot locate utils/div_free_noise.py relative to cond_diffusion.py"
    )
_df_spec = importlib.util.spec_from_file_location("div_free_noise", _df_path)
_df_mod  = importlib.util.module_from_spec(_df_spec)
_df_spec.loader.exec_module(_df_mod)

NOISE_TYPES            = _df_mod.NOISE_TYPES
_divergence_free_noise = _df_mod.divergence_free_noise


class CondDDPM:
    """
    Denoising Diffusion Probabilistic Model utilities for conditioned models.

    All forward calls pass a conditioning tensor `cond` to the model:
        model(xt, t, cond)   →   predicted_noise

    Noise schedule
    --------------
    Cosine (default) or linear.  Same formulation as the base DDPM.

    Training loss
    -------------
    Epsilon-prediction MSE on ocean pixels only:
        L = MSE( ε_pred * ocean_mask,  ε_true * ocean_mask )

    Reverse sampling
    ----------------
    p_sample_step : single DDPM reverse step
    sample        : full reverse chain from T-step Gaussian noise to x0
    """

    def __init__(
        self,
        T:               int                 = 1000,
        beta_schedule:   str                 = "cosine",
        device:          str                 = "cpu",
        noise_scale:     float               = 1.0,
        noise_type:      str                 = "gaussian",
        spectral_filter: torch.Tensor | None = None,
    ):
        self.T           = T
        self.device      = device
        self.noise_scale = noise_scale

        if noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got '{noise_type}'")
        self.noise_type = noise_type

        # Spectral filter for colored div-free noise (CPU tensor, or None)
        if spectral_filter is not None:
            self.spectral_filter = spectral_filter.cpu().float()
        else:
            self.spectral_filter = None

        betas = self._cosine_betas(T) if beta_schedule == "cosine" \
                else torch.linspace(1e-4, 0.02, T)

        self.betas         = betas.to(device)
        alphas             = 1.0 - self.betas
        self.alphas        = alphas
        self.alpha_bar     = torch.cumprod(alphas, dim=0)
        # alpha_bar_prev[t] = ᾱ_{t-1}, with alpha_bar_prev[0] = 1
        self.alpha_bar_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bar[:-1]]
        )
        self.sqrt_ab       = self.alpha_bar.sqrt()
        self.sqrt_one_mab  = (1.0 - self.alpha_bar).sqrt()

    # ------------------------------------------------------------------
    # Noise sampler
    # ------------------------------------------------------------------

    def _sample_noise(self, like: torch.Tensor) -> torch.Tensor:
        """Return noise (gaussian or divergence-free) shaped like `like`."""
        if self.noise_type == "gaussian":
            return torch.randn_like(like)
        return _divergence_free_noise(
            like.shape,
            device=str(like.device),
            spectral_filter=self.spectral_filter,
        )

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _cosine_betas(self, T: int, s: float = 0.008) -> torch.Tensor:
        steps = T + 1
        t  = torch.linspace(0, T, steps) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        return (1.0 - ab[1:] / ab[:-1]).clamp(0, 0.999)

    # ------------------------------------------------------------------
    # Forward process  q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) = N(√ᾱ_t · x0, (1−ᾱ_t) · noise_scale² · I)."""
        if noise is None:
            noise = self._sample_noise(x0) * self.noise_scale
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
        cond:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Epsilon-prediction MSE loss, ocean pixels only.

        Args:
            model:     CondUNet — forward(xt, t, cond) → predicted noise
            x0:        (B, 2, H, W) clean velocity fields
            land_mask: (H, W) bool, True = land (excluded from loss)
            cond:      (B, cond_in_ch, H, W) conditioning map

        Returns:
            loss: scalar tensor
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t, cond)

        ocean = (~land_mask).float()[None, None]   # (1, 1, H, W)
        return F.mse_loss(pred_noise * ocean, noise * ocean)

    # ------------------------------------------------------------------
    # Reverse process  p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(
        self,
        model: torch.nn.Module,
        xt:    torch.Tensor,
        t_int: int,
        cond:  torch.Tensor,
    ) -> torch.Tensor:
        """
        One DDPM reverse step with conditioning.

        Args:
            model: CondUNet
            xt:    (B, 2, H, W) current noisy field
            t_int: integer timestep (same for whole batch)
            cond:  (B, cond_in_ch, H, W) conditioning map

        Returns:
            x_{t-1}: (B, 2, H, W)
        """
        B = xt.shape[0]
        t = torch.full((B,), t_int, device=self.device, dtype=torch.long)

        pred_noise = model(xt, t, cond)

        ab      = self.alpha_bar[t_int]
        ab_prev = self.alpha_bar_prev[t_int]
        beta    = self.betas[t_int]
        alpha   = self.alphas[t_int]

        # Predicted x0 — clamp to ±3σ of the data (noise_scale ≈ data std)
        x0_pred = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-3.0 * self.noise_scale, 3.0 * self.noise_scale)

        if t_int == 0:
            return x0_pred

        # DDPM posterior mean
        mean = (
            (ab_prev.sqrt() * beta / (1.0 - ab)) * x0_pred +
            (alpha.sqrt() * (1.0 - ab_prev) / (1.0 - ab)) * xt
        )
        var = beta * (1.0 - ab_prev) / (1.0 - ab)
        return mean + var.sqrt() * self.noise_scale * self._sample_noise(xt)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        cond:  torch.Tensor,
        shape: tuple | None = None,
    ) -> torch.Tensor:
        """
        Full reverse diffusion from Gaussian noise to x0, conditioned on `cond`.

        Args:
            model: CondUNet
            cond:  (B, cond_in_ch, H, W) conditioning map (fixed throughout)
            shape: output shape; defaults to (B, 2, 94, 44)

        Returns:
            x0_pred: (B, 2, H, W)
        """
        if shape is None:
            B = cond.shape[0]
            shape = (B, 2, 94, 44)
        xt = self._sample_noise(torch.empty(*shape, device=self.device)) * self.noise_scale
        for t in reversed(range(self.T)):
            xt = self.p_sample_step(model, xt, t, cond)
        return xt

    # ------------------------------------------------------------------
    # Forward step from x_{t-1}  (used by RePaint resampling)
    # ------------------------------------------------------------------

    def q_sample_from_prev(self, x_prev: torch.Tensor, t_int: int) -> torch.Tensor:
        """
        One forward noising step: x_t ~ q(x_t | x_{t-1}).
            x_t = sqrt(alpha_t) * x_{t-1} + sqrt(beta_t) * noise_scale * eps
        """
        alpha = self.alphas[t_int]
        beta  = self.betas[t_int]
        return alpha.sqrt() * x_prev + beta.sqrt() * self.noise_scale * self._sample_noise(x_prev)

    # ------------------------------------------------------------------
    # RePaint inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def repaint(
        self,
        model:      torch.nn.Module,
        cond:       torch.Tensor,       # (1, cond_in_ch, H, W)
        x0_known:   torch.Tensor,       # (1, 2, H, W) true u/v at path cells, 0 elsewhere
        path_mask:  torch.Tensor,       # (1, 1, H, W) float, 1 = known cell
        ocean_mask: torch.Tensor,       # (1, 1, H, W) float, 1 = ocean cell
        r:          int = 10,
    ) -> torch.Tensor:
        """
        RePaint reverse diffusion with known path cells anchored at each step.

        At every timestep t, for r iterations:
          1. Model reverse step  → x_{t-1} (unknown pixels)
          2. Forward-diffuse x0_known to t-1  → x_{t-1} (known pixels)
          3. Merge known / unknown
          4. If not last iteration: resample forward to t and repeat

        Args:
            model:      CondUNet
            cond:       (1, cond_in_ch, H, W) conditioning tensor
            x0_known:   (1, 2, H, W) ground-truth u/v at path cells, 0 elsewhere
            path_mask:  (1, 1, H, W) float — 1 at known cells
            ocean_mask: (1, 1, H, W) float — 1 at ocean cells
            r:          resampling iterations per timestep (r=10 is RePaint default)

        Returns:
            x0_pred: (1, 2, H, W)
        """
        B, _, H, W = x0_known.shape
        xt = self._sample_noise(torch.empty(B, 2, H, W, device=self.device)) * self.noise_scale * ocean_mask

        for t_int in reversed(range(self.T)):
            for j in range(r):
                # Step 1: model reverse step
                xt_model = self.p_sample_step(model, xt, t_int, cond)

                # Step 2: forward-diffuse x0_known to t-1
                t_prev = max(t_int - 1, 0)
                t_prev_t = torch.full((B,), t_prev, device=self.device, dtype=torch.long)
                xt_known, _ = self.q_sample(x0_known, t_prev_t)

                # Step 3: merge
                xt = path_mask * xt_known + (1.0 - path_mask) * xt_model
                xt = xt * ocean_mask

                # Step 4: resample forward if not last iteration
                if j < r - 1 and t_int > 0:
                    xt = self.q_sample_from_prev(xt, t_int)
                    xt = xt * ocean_mask

        return xt
