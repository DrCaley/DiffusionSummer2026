"""
Multi-Loss DDPM — extends the base DDPM with switchable structural loss terms.

Supported loss modes (set via --loss at training time):

  eps          Pure epsilon-MSE only. Identical to Basic DDPM.
  curl_div     epsilon-MSE + curl/divergence penalty on reconstructed x̂₀.
               Equivalent to the Topo DDPM.
  spectral     epsilon-MSE + FFT power-spectrum penalty on reconstructed x̂₀.
               Penalises differences in multi-scale energy distribution.
  okubo_weiss  epsilon-MSE + Okubo-Weiss parameter penalty on reconstructed x̂₀.
               W = sₙ² + s_s² − ω²  classifies eddy (W<0) vs strain (W>0)
               regions; matching W reproduces eddy boundaries.
  wasserstein  epsilon-MSE + Sinkhorn–Wasserstein distance between the
               vorticity fields of x̂₀ and x₀, treated as 2-D point clouds
               weighted by |ω|.  Penalises eddies in the wrong location
               rather than wrong amplitude.  Requires geomloss.

All structural terms are:
  - computed on the denoised reconstruction  x̂₀ = (xₜ − √(1−ᾱₜ)·ε̂) / √ᾱₜ
  - masked to ocean pixels only
  - returned separately so train.py can log them
"""

import importlib.util
import os

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Load base DDPM from the sibling 'Basic DDPM' folder via importlib so that
# Python's module cache never conflates this diffusion.py with the base one.
# _patch_server.py rewrites the path tuple for the flat server layout.
# ---------------------------------------------------------------------------
_base_path = os.path.join(
    os.path.dirname(__file__), "..", "Basic DDPM", "diffusion.py"
)
_spec = importlib.util.spec_from_file_location(
    "basic_ddpm_diffusion", os.path.abspath(_base_path)
)
_base_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_mod)

DDPM = _base_mod.DDPM

LOSS_MODES = ("eps", "curl_div", "spectral", "okubo_weiss", "wasserstein")


class MultiLossDDPM(DDPM):
    """
    DDPM subclass with one or more switchable auxiliary structural losses.

    Args:
        loss_type:     A loss name or list of loss names from LOSS_MODES.
                       'eps' disables all auxiliary terms.
                       Multiple names can be combined: ['spectral', 'okubo_weiss'].
        weights:       Dict mapping each loss name to its scalar weight λ.
                       e.g. {'spectral': 0.0002, 'okubo_weiss': 0.001}
                       Each term contributes independently: total = eps + λ₁·L₁ + λ₂·L₂ + …
        sinkhorn_blur: Entropic regularisation blur for Sinkhorn.
                       Only used when 'wasserstein' is in loss_type.
    """

    def __init__(
        self,
        T:              int                       = 1000,
        beta_schedule:  str                       = "cosine",
        device:         str                       = "cpu",
        loss_type:      str | list[str]           = "spectral",
        weights:        dict[str, float] | None   = None,
        sinkhorn_blur:  float                     = 0.05,
    ):
        super().__init__(T=T, beta_schedule=beta_schedule, device=device)

        # Normalise to a list and validate every entry
        if isinstance(loss_type, str):
            loss_type = [loss_type]
        for lt in loss_type:
            if lt not in LOSS_MODES:
                raise ValueError(f"loss_type must be one of {LOSS_MODES}, got '{lt}'")
        self.loss_types    = loss_type
        self.sinkhorn_blur = sinkhorn_blur

        # Build weights dict — default to 1.0 for any loss not explicitly specified
        self.weights: dict[str, float] = {lt: 1.0 for lt in loss_type}
        if weights is not None:
            self.weights.update(weights)

        # Lazy-import geomloss only when needed so the other modes don't
        # require the package to be installed.
        if "wasserstein" in self.loss_types:
            try:
                from geomloss import SamplesLoss
                self._sinkhorn = SamplesLoss(
                    loss="sinkhorn", p=1, blur=sinkhorn_blur,
                    scaling=0.5, backend="tensorized",
                )
            except ImportError as e:
                raise ImportError(
                    "loss_type='wasserstein' requires geomloss.  "
                    "Install it with: pip install geomloss"
                ) from e

    # ------------------------------------------------------------------
    # Shared finite-difference helper
    # ------------------------------------------------------------------

    def _jacobian(self, field: torch.Tensor) -> tuple:
        """
        Compute all four first-order spatial derivatives of a (B, 2, H, W)
        vector field using central-difference convolution.

        Returns: (du_dx, du_dy, dv_dx, dv_dy)  each (B, 1, H, W)
        """
        u = field[:, 0:1]
        v = field[:, 1:2]

        kx = torch.tensor(
            [[[[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]]]],
            device=field.device,
        ) / 2.0

        ky = torch.tensor(
            [[[[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]]]],
            device=field.device,
        ) / 2.0

        du_dx = F.conv2d(u, kx, padding=1)
        du_dy = F.conv2d(u, ky, padding=1)
        dv_dx = F.conv2d(v, kx, padding=1)
        dv_dy = F.conv2d(v, ky, padding=1)

        return du_dx, du_dy, dv_dx, dv_dy

    # ------------------------------------------------------------------
    # Auxiliary loss implementations
    # ------------------------------------------------------------------

    def _curl_div_features(self, field: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, 2, H, W): channel 0 = curl (ω = dv/dx − du/dy),
                               channel 1 = divergence (D = du/dx + dv/dy).
        """
        du_dx, du_dy, dv_dx, _ = self._jacobian(field)
        _, _, dv_dx2, dv_dy = self._jacobian(field)
        du_dx, du_dy, dv_dx, dv_dy = self._jacobian(field)

        curl = dv_dx - du_dy
        div  = du_dx + dv_dy
        return torch.cat([curl, div], dim=1)

    def _spectral_features(self, field: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, 2, H, W//2+1) power spectra of both velocity components.
        Uses rfft2 so the output is real (magnitude).
        """
        u = field[:, 0]   # (B, H, W)
        v = field[:, 1]

        Su = torch.fft.rfft2(u).abs()   # (B, H, W//2+1)
        Sv = torch.fft.rfft2(v).abs()

        return torch.stack([Su, Sv], dim=1)   # (B, 2, H, W//2+1)

    def _okubo_weiss(self, field: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, 1, H, W) Okubo-Weiss parameter W:

            sₙ = du/dx − dv/dy   (normal strain)
            s_s = du/dy + dv/dx  (shear strain)
            ω  = dv/dx − du/dy   (vorticity / curl)
            W  = sₙ² + s_s² − ω²

        W < 0  →  rotation-dominated (inside eddies)
        W > 0  →  strain-dominated (between eddies)
        """
        du_dx, du_dy, dv_dx, dv_dy = self._jacobian(field)

        sn = du_dx - dv_dy          # normal strain
        ss = du_dy + dv_dx          # shear strain
        w  = dv_dx - du_dy          # vorticity

        W = sn**2 + ss**2 - w**2    # (B, 1, H, W)
        return W

    def _vorticity_cloud(self, field: torch.Tensor, ocean: torch.Tensor) -> torch.Tensor:
        """
        Convert the vorticity field into a weighted 2-D point cloud for
        use with geomloss.SamplesLoss.

        The vorticity is split into positive (cyclonic) and negative
        (anticyclonic) parts separately so the clouds have non-negative
        weights; both are concatenated into a single cloud.

        Args:
            field: (B, 2, H, W)
            ocean: (1, 1, H, W) float mask, 1 = ocean

        Returns:
            points:  (B, N, 2) — normalised (row, col) coordinates in [0,1]
            weights: (B, N, 1) — normalised |ω| weights summing to 1 per sample

        N = H * W  (land pixels get weight 0 and are kept for simplicity;
        geomloss handles near-zero weights gracefully).
        """
        du_dx, du_dy, dv_dx, dv_dy = self._jacobian(field)
        curl = (dv_dx - du_dy) * ocean   # (B, 1, H, W)

        B, _, H, W = curl.shape

        # Build a regular (row, col) coordinate grid normalised to [0, 1]
        rows = torch.linspace(0, 1, H, device=field.device)
        cols = torch.linspace(0, 1, W, device=field.device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")   # (H, W)
        coords = torch.stack([grid_r, grid_c], dim=-1)               # (H, W, 2)
        coords = coords.view(1, H * W, 2).expand(B, -1, -1)          # (B, N, 2)

        # Weights = |ω|, normalised so they sum to 1 per sample
        weights = curl.abs().view(B, H * W)                          # (B, N)
        # Add a small epsilon to avoid division by zero for near-zero fields
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        weights = weights.unsqueeze(-1)                              # (B, N, 1)

        return coords, weights

    def _wasserstein_loss(self, x0_pred: torch.Tensor, x0_true: torch.Tensor,
                          ocean: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn–Wasserstein distance between the vorticity point clouds of
        x0_pred and x0_true, averaged over the batch.
        """
        coords_pred, w_pred = self._vorticity_cloud(x0_pred, ocean)
        coords_true, w_true = self._vorticity_cloud(x0_true, ocean)

        # geomloss expects: loss(α_weights, α_points, β_weights, β_points)
        # weights must sum to 1 (already done above).
        # squeeze last dim of weights: geomloss wants (B, N) for the weight arg
        dist = self._sinkhorn(
            w_pred.squeeze(-1), coords_pred,
            w_true.squeeze(-1), coords_true,
        )   # (B,)
        return dist.mean()

    # ------------------------------------------------------------------
    # Combined training loss
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model:     torch.nn.Module,
        x0:        torch.Tensor,
        land_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute epsilon-MSE + each active auxiliary loss with its own weight.

        Args:
            model:     UNet noise predictor
            x0:        (B, 2, H, W) clean fields
            land_mask: (H, W) bool, True = land

        Returns:
            (total_loss, eps_loss, indiv)
            indiv: dict mapping each active loss name to its unweighted value.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)

        ocean = (~land_mask).float()[None, None]   # (1, 1, H, W)

        # --- Epsilon loss ---
        eps_loss = F.mse_loss(pred_noise * ocean, noise * ocean)

        if self.loss_types == ["eps"]:
            return eps_loss, eps_loss, {}

        # --- Reconstruct x̂₀ ---
        sqrt_ab  = self.sqrt_ab[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_mab[t][:, None, None, None]
        x0_pred  = (xt - sqrt_mab * pred_noise) / sqrt_ab.clamp(min=1e-8)

        # --- Compute each active auxiliary loss independently ---
        indiv: dict[str, torch.Tensor] = {}

        for lt in self.loss_types:
            if lt == "eps":
                continue

            elif lt == "curl_div":
                feat_pred = self._curl_div_features(x0_pred) * ocean
                feat_true = self._curl_div_features(x0)       * ocean
                indiv[lt] = F.mse_loss(feat_pred, feat_true)

            elif lt == "spectral":
                feat_pred = self._spectral_features(x0_pred * ocean)
                feat_true = self._spectral_features(x0       * ocean)
                indiv[lt] = F.mse_loss(feat_pred, feat_true)

            elif lt == "okubo_weiss":
                feat_pred = self._okubo_weiss(x0_pred) * ocean
                feat_true = self._okubo_weiss(x0)       * ocean
                indiv[lt] = F.mse_loss(feat_pred, feat_true)

            elif lt == "wasserstein":
                indiv[lt] = self._wasserstein_loss(x0_pred, x0_true=x0, ocean=ocean)

        # total = eps + sum(lambda_i * L_i)
        total = eps_loss + sum(self.weights[lt] * v for lt, v in indiv.items())
        return total, eps_loss, indiv
