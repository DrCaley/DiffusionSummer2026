import importlib.util
import math
import os

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Load loss_functions.py from Model Parameters/ via importlib so this file
# can be imported from any working directory.
# Searches up to 3 levels above this file's location so the same diffusion.py
# works both when installed under DDPM/model/ (local) and at the repo root
# (remote server with flat structure).
# ---------------------------------------------------------------------------
_lf_path = None
for _up in range(4):
    _candidate = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        *(['..'] * _up),
        "Model Parameters",
        "loss_functions.py",
    ))
    if os.path.isfile(_candidate):
        _lf_path = _candidate
        break
if _lf_path is None:
    raise FileNotFoundError(
        "Cannot locate loss_functions.py under any Model Parameters/ "
        "directory relative to diffusion.py"
    )
_lf_spec = importlib.util.spec_from_file_location(
    "loss_functions", os.path.abspath(_lf_path)
)
_lf_mod = importlib.util.module_from_spec(_lf_spec)
_lf_spec.loader.exec_module(_lf_mod)

LOSS_MODES    = _lf_mod.LOSS_MODES
DEFAULT_WEIGHTS = _lf_mod.DEFAULT_WEIGHTS


class DDPM:
    """
    Denoising Diffusion Probabilistic Model utilities.

    Handles the cosine noise schedule, forward process q(x_t | x_0),
    training loss (with optional structural regularisation), and a
    single reverse step p(x_{t-1} | x_t).

    Loss modes (set via loss_types):
        eps          Pure epsilon-MSE only (default).
        curl_div     + curl/divergence penalty on reconstructed x̂₀.
        spectral     + FFT power-spectrum penalty on reconstructed x̂₀.
        okubo_weiss  + Okubo-Weiss eddy-structure penalty.
        wasserstein  + Sinkhorn-Wasserstein vorticity distance (needs geomloss).

    Multiple modes can be combined: loss_types=["spectral", "okubo_weiss"].
    Each has its own independent weight from the weights dict.
    """

    def __init__(
        self,
        T:             int                      = 1000,
        beta_schedule: str                      = "cosine",
        device:        str                      = "cpu",
        loss_types:    str | list[str]          = "eps",
        weights:       dict[str, float] | None  = None,
        sinkhorn_blur: float                    = 0.05,
    ):
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

        # --- Loss configuration ---
        if isinstance(loss_types, str):
            loss_types = [loss_types]
        for lt in loss_types:
            if lt not in LOSS_MODES:
                raise ValueError(f"loss_types must be from {LOSS_MODES}, got '{lt}'")
        self.loss_types = loss_types

        # Per-loss weights: start from defaults, then apply any overrides
        self.weights: dict[str, float] = {
            lt: DEFAULT_WEIGHTS.get(lt, 1.0) for lt in loss_types if lt != "eps"
        }
        if weights is not None:
            self.weights.update(weights)

        # Lazy-load geomloss only when wasserstein is requested
        self._sinkhorn = None
        if "wasserstein" in self.loss_types:
            try:
                from geomloss import SamplesLoss
                self._sinkhorn = SamplesLoss(
                    loss="sinkhorn", p=1, blur=sinkhorn_blur,
                    scaling=0.5, backend="tensorized",
                )
            except ImportError as e:
                raise ImportError(
                    "loss_types='wasserstein' requires geomloss.  "
                    "Install it with: pip install geomloss"
                ) from e

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
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Epsilon-prediction MSE loss plus any active auxiliary structural losses.

        Args:
            model:     UNet that predicts noise
            x0:        (B, 2, H, W) clean fields
            land_mask: (H, W) bool, True = land (excluded from loss)

        Returns:
            (total, eps_loss, indiv)
            total:    scalar loss used for backprop
            eps_loss: the epsilon-MSE component alone
            indiv:    dict of {loss_name: unweighted_value} for each aux loss
                      (empty dict when loss_types == ["eps"])
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)

        ocean = (~land_mask).float()[None, None]   # (1, 1, H, W)
        eps_loss = F.mse_loss(pred_noise * ocean, noise * ocean)

        if self.loss_types == ["eps"]:
            return eps_loss, eps_loss, {}

        # Reconstruct x̂₀
        sqrt_ab  = self.sqrt_ab[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_mab[t][:, None, None, None]
        x0_pred  = (xt - sqrt_mab * pred_noise) / sqrt_ab.clamp(min=1e-8)

        # Compute each active auxiliary loss
        indiv: dict[str, torch.Tensor] = {}
        for lt in self.loss_types:
            if lt == "eps":
                continue
            elif lt == "curl_div":
                indiv[lt] = _lf_mod.curl_div_loss(x0_pred, x0, ocean)
            elif lt == "spectral":
                indiv[lt] = _lf_mod.spectral_loss(x0_pred, x0, ocean)
            elif lt == "okubo_weiss":
                indiv[lt] = _lf_mod.okubo_weiss_loss(x0_pred, x0, ocean)
            elif lt == "wasserstein":
                indiv[lt] = _lf_mod.wasserstein_loss(
                    x0_pred, x0, ocean, self._sinkhorn
                )

        total = eps_loss + sum(self.weights[lt] * v for lt, v in indiv.items())
        return total, eps_loss, indiv

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
