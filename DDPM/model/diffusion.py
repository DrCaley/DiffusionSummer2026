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
# Search for loss_functions.py at several relative depths. It may live directly
# under "Model Parameters/" (older layout) or under the "Loss Function/"
# subfolder (after the repo reorganisation).
_lf_subpaths = (
    ("Model Parameters", "loss_functions.py"),
    ("Model Parameters", "Loss Function", "loss_functions.py"),
)
for _up in range(4):
    for _sub in _lf_subpaths:
        _candidate = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            *(['..'] * _up),
            *_sub,
        ))
        if os.path.isfile(_candidate):
            _lf_path = _candidate
            break
    if _lf_path is not None:
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

LOSS_MODES      = _lf_mod.LOSS_MODES
DEFAULT_WEIGHTS = _lf_mod.DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------
# Load div_free_noise.py from utils/ via importlib.
# ---------------------------------------------------------------------------
_df_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "utils", "div_free_noise.py"
)
_df_spec = importlib.util.spec_from_file_location(
    "div_free_noise", os.path.abspath(_df_path)
)
_df_mod = importlib.util.module_from_spec(_df_spec)
_df_spec.loader.exec_module(_df_mod)

NOISE_TYPES           = _df_mod.NOISE_TYPES
_divergence_free_noise = _df_mod.divergence_free_noise


class DDPM:
    """
    Denoising Diffusion Probabilistic Model utilities.

    Handles the cosine noise schedule, forward process q(x_t | x_0),
    training loss (with optional structural regularisation), and a
    single reverse step p(x_{t-1} | x_t).

    Loss modes (set via loss_types):
        eps          Pure epsilon-MSE only (default).
        angle        Directional (cosine) loss on x̂₀ — penalises only the angle
                     between predicted and true velocity vectors (magnitude
                     ignored). Use alone for a pure flow-direction model.
        curl_div     curl/divergence penalty on reconstructed x̂₀.
        spectral     FFT power-spectrum penalty on reconstructed x̂₀.
        okubo_weiss  Okubo-Weiss eddy-structure penalty.
        wasserstein  Sinkhorn-Wasserstein vorticity distance (needs geomloss).
        stream_function  stream-function (Poisson-solve) penalty.
        strain_rate  strain-rate tensor invariants penalty.

    Multiple modes can be combined: loss_types=["spectral", "okubo_weiss"].
    Omitting "eps" trains with *only* the auxiliary losses (no MSE term).
    Each auxiliary loss has its own independent weight from the weights dict.
    """

    def __init__(
        self,
        T:                  int                      = 1000,
        beta_schedule:      str                      = "cosine",
        device:             str                      = "cpu",
        noise_type:         str                      = "gaussian",
        loss_types:         str | list[str]          = "eps",
        weights:            dict[str, float] | None  = None,
        sinkhorn_blur:      float                    = 0.05,
        spectral_filter:    torch.Tensor | None      = None,
        noise_scale:        float                    = 1.0,
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
    # Noise sampler
    # ------------------------------------------------------------------

    def _sample_noise(self, like: torch.Tensor) -> torch.Tensor:
        """Return noise with the same shape/device as `like`."""
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
        """Sample x_t ~ q(x_t | x_0) = N(sqrt(ᾱ_t)*x0, (1-ᾱ_t)*noise_scale²*I)."""
        if noise is None:
            noise = self._sample_noise(x0) * self.noise_scale
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
            elif lt == "angle":
                indiv[lt] = _lf_mod.angle_loss(x0_pred, x0, ocean)
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
            elif lt == "stream_function":
                indiv[lt] = _lf_mod.stream_function_loss(x0_pred, x0, ocean)
            elif lt == "strain_rate":
                indiv[lt] = _lf_mod.strain_rate_loss(x0_pred, x0, ocean)

        aux_total = sum(self.weights[lt] * v for lt, v in indiv.items())
        # Only add epsilon-MSE when "eps" is explicitly listed
        total = (eps_loss + aux_total) if "eps" in self.loss_types else aux_total
        return total, eps_loss, indiv

    # ------------------------------------------------------------------
    # Training — stream-function (divergence-free) calibrated model
    # ------------------------------------------------------------------

    def training_loss_streamfn(
        self,
        model:         torch.nn.Module,
        x0:            torch.Tensor,
        land_mask:     torch.Tensor,
        lambda_angle:  float = 1.0,
        min_snr_gamma: float = 5.0,
        cond:          torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Calibrated x0-prediction loss for the stream-function model.

        The model (StreamFunctionUNet) returns x̂₀ directly as a divergence-free
        field (the curl of its scalar stream function).  It is supervised with a
        Min-SNR-γ–weighted reconstruction MSE plus an angle (direction) term:

            L = w_t · ‖x̂₀ − x₀‖²_ocean  +  λ · (1 − cosθ)_ocean

        where  w_t = min(SNR_t, γ) / mean(min(SNR_t, γ))  is the Min-SNR-γ weight
        (Hang et al., 2023).  Clamping the SNR recovers the noise-level balancing
        that motivates v-prediction while keeping the stream-function (x0-space)
        parameterisation that guarantees incompressibility.  The angle term keeps
        flow direction — the eddy-detection north star — first-class.

        Args:
            model:         StreamFunctionUNet predicting the clean field x̂₀.
            x0:            (B, 2, H, W) clean fields.
            land_mask:     (H, W) bool, True = land (excluded from loss).
            lambda_angle:  weight λ on the directional (1−cosθ) term.
            min_snr_gamma: Min-SNR clamp γ (paper default 5.0).
            cond:          (B, C, H, W) optional conditioning channels passed to
                           the model (for the conditional stream-function variant).
                           None for the unconditional model.

        Returns:
            (total, recon_mse, indiv)
            total:     scalar loss used for backprop
            recon_mse: unweighted masked reconstruction MSE (for logging)
            indiv:     {"angle": value}
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, _ = self.q_sample(x0, t)
        x0_pred = model(xt, t) if cond is None else model(xt, t, cond)   # div-free x̂₀

        ocean = (~land_mask).float()[None, None]     # (1, 1, H, W)
        denom = (ocean.sum() * B).clamp(min=1.0)

        # Min-SNR-γ timestep weight (per sample)
        ab  = self.alpha_bar[t]
        snr = ab / (1.0 - ab).clamp(min=1e-8)
        w   = snr.clamp(max=min_snr_gamma)
        w   = (w / w.mean().clamp(min=1e-8))[:, None, None, None]

        se        = ((x0_pred - x0) * ocean) ** 2     # (B, 2, H, W)
        recon_mse = se.sum() / denom                  # unweighted (logging)
        weighted  = (w * se).sum() / denom            # Min-SNR weighted (backprop)

        ang   = _lf_mod.angle_loss(x0_pred, x0, ocean)
        total = weighted + lambda_angle * ang
        return total, recon_mse, {"angle": ang}


    # ------------------------------------------------------------------
    # Inference schedule helper
    # ------------------------------------------------------------------

    def build_inference_schedule(self, n_steps: int) -> list[tuple[int, int]]:
        """
        Build a subsampled list of (t, t_prev) integer pairs for inference.

        The T training timesteps are evenly divided into n_steps intervals.
        Iterating in reverse gives the denoising order.

        Args:
            n_steps: number of reverse steps (≤ T).  If n_steps == T the full
                     schedule is returned.  Must divide evenly or the nearest
                     integer spacing is used.

        Returns:
            List of (t, t_prev) pairs in *reverse* order (from t=T-1 down to 0),
            e.g. for T=1000, n_steps=100:
              [(999, 989), (989, 979), ..., (19, 9), (9, -1)]
            t_prev == -1 signals the final step (return x̂₀ directly).
        """
        step_size = self.T // n_steps
        # Timesteps in reverse: T-1, T-1-step, ..., step-1, 0  (roughly)
        ts = list(reversed(range(step_size - 1, self.T, step_size)))
        pairs = [(ts[i], ts[i + 1] if i + 1 < len(ts) else -1)
                 for i in range(len(ts))]
        return pairs

    # ------------------------------------------------------------------
    # Single reverse step  p(x_{t_prev} | x_t)  — supports subsampled schedules
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(
        self,
        model:     torch.nn.Module,
        xt:        torch.Tensor,
        t_int:     int,
        t_prev_int: int = -1,
    ) -> torch.Tensor:
        """
        One DDPM reverse step, supporting non-consecutive (subsampled) schedules.

        Args:
            model:      trained UNet
            xt:         (B, 2, H, W) current noisy field
            t_int:      current integer timestep
            t_prev_int: previous integer timestep (-1 means final step → return x̂₀)
        Returns:
            x_{t_prev}: (B, 2, H, W)
        """
        B = xt.shape[0]
        t = torch.full((B,), t_int, device=self.device, dtype=torch.long)

        pred_noise = model(xt, t)

        ab  = self.alpha_bar[t_int]

        # Predicted x0 — clamp to ±3σ of the data (noise_scale ≈ data std)
        x0_pred = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-3.0 * self.noise_scale, 3.0 * self.noise_scale)

        if t_prev_int < 0:
            return x0_pred

        ab_prev = self.alpha_bar[t_prev_int]

        # Effective β for this (possibly multi-step) interval:
        #   β_eff = 1 - ᾱ_{t_prev} / ᾱ_t   (always positive: ab_prev > ab)
        # Posterior variance (general DDPM, works for any step size):
        #   σ² = (1 - ᾱ_{t_prev}) / (1 - ᾱ_t) * β_eff
        beta_eff = 1.0 - ab / ab_prev
        var      = (1.0 - ab_prev) / (1.0 - ab) * beta_eff

        # Posterior mean coefficients (DDPM, general step)
        coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
        coef2 = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
        mean  = coef1 * x0_pred + coef2 * xt

        return mean + var.sqrt() * self.noise_scale * self._sample_noise(xt)

    # ------------------------------------------------------------------
    # Forward jump  q(x_t | x_{t_prev})  — used by RePaint resampling
    # ------------------------------------------------------------------

    def q_sample_from_prev(
        self,
        x_prev:    torch.Tensor,
        t_int:     int,
        t_prev_int: int = -1,
    ) -> torch.Tensor:
        """
        Jump forward from x_{t_prev} to x_t using the marginal.
        Works for arbitrary (non-consecutive) step pairs.

        x_t = sqrt(ᾱ_t / ᾱ_{t_prev}) * x_{t_prev}
              + sqrt(1 - ᾱ_t / ᾱ_{t_prev}) * noise_scale * eps
        """
        ab_t = self.alpha_bar[t_int]
        if t_prev_int < 0:
            return ab_t.sqrt() * x_prev + (1.0 - ab_t).sqrt() * self.noise_scale * self._sample_noise(x_prev)
        ab_prev = self.alpha_bar[t_prev_int]
        ratio   = ab_t / ab_prev
        return ratio.sqrt() * x_prev + (1.0 - ratio).sqrt() * self.noise_scale * self._sample_noise(x_prev)


# ---------------------------------------------------------------------------
# Inference adapter: stream-function (x0) model → epsilon-equivalent
# ---------------------------------------------------------------------------

class EpsFromStreamFn(torch.nn.Module):
    """
    Wrap a stream-function (x0-prediction) model so it exposes an
    *epsilon-equivalent* output, letting it drop into any sampler that expects
    an eps-predicting network (p_sample_step, RePaint, PPR, DPS) with no other
    change.

        x̂₀(x_t, t) = stream_model(x_t, t)               (divergence-free field)
        ε̂(x_t, t) = (x_t − √ᾱ_t · x̂₀) / √(1 − ᾱ_t)

    Any sampler that reconstructs x̂₀ = (x_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t recovers
    exactly the divergence-free model field, so divergence-free structure is
    preserved through the reverse process.
    """

    def __init__(self, stream_model: torch.nn.Module, diffusion: "DDPM",
                 cond: torch.Tensor | None = None):
        super().__init__()
        self.stream_model = stream_model
        self._ab = diffusion.alpha_bar
        # Optional fixed conditioning (obs + temporal priors + geometry).  It is
        # constant across the reverse process for a given sample, so it is set
        # once here and threaded into every model call.  None => unconditional.
        self.cond = cond

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.cond is None:
            x0 = self.stream_model(xt, t)
        else:
            cond = self.cond
            if cond.shape[0] != xt.shape[0]:           # broadcast to batch
                cond = cond.expand(xt.shape[0], *cond.shape[1:])
            x0 = self.stream_model(xt, t, cond)
        ab = self._ab[t][:, None, None, None]
        sqrt_ab  = ab.sqrt()
        sqrt_mab = (1.0 - ab).sqrt().clamp(min=1e-8)
        return (xt - sqrt_ab * x0) / sqrt_mab

