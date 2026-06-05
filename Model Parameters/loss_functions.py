"""
Structural auxiliary loss functions for ocean current diffusion models.

All functions are stateless and operate on (B, 2, H, W) vector field tensors.
Import this module from any training script to add structural regularisation
on top of the base epsilon-MSE loss.

Available loss modes (LOSS_MODES):
    eps          Pure epsilon-MSE only — no auxiliary term.
    curl_div     MSE between curl and divergence fields of x̂₀ and x₀.
    spectral     MSE between FFT power spectra of x̂₀ and x₀.
    okubo_weiss  MSE between Okubo-Weiss parameters of x̂₀ and x₀.
    wasserstein  Sinkhorn–Wasserstein distance between vorticity point clouds.

Default weights (DEFAULT_WEIGHTS):
    curl_div     0.0002
    spectral     0.0002
    okubo_weiss  0.001
    wasserstein  1.0
"""

import torch
import torch.nn.functional as F

LOSS_MODES = ("eps", "curl_div", "spectral", "okubo_weiss", "wasserstein")

DEFAULT_WEIGHTS: dict[str, float] = {
    "curl_div":    0.0002,
    "spectral":    0.0002,
    "okubo_weiss": 0.001,
    "wasserstein": 1.0,
}


# ---------------------------------------------------------------------------
# Shared finite-difference helper
# ---------------------------------------------------------------------------

def _jacobian(field: torch.Tensor):
    """
    First-order spatial derivatives of a (B, 2, H, W) vector field via
    central-difference convolution.

    Returns: (du_dx, du_dy, dv_dx, dv_dy)  each (B, 1, H, W).
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

    return (
        F.conv2d(u, kx, padding=1),   # du_dx
        F.conv2d(u, ky, padding=1),   # du_dy
        F.conv2d(v, kx, padding=1),   # dv_dx
        F.conv2d(v, ky, padding=1),   # dv_dy
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def curl_div_loss(
    pred:  torch.Tensor,
    true:  torch.Tensor,
    ocean: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between the curl and divergence of pred and true, masked to ocean.

    Args:
        pred, true: (B, 2, H, W) vector fields
        ocean:      (1, 1, H, W) float mask (1 = ocean, 0 = land)
    """
    def _features(field):
        du_dx, du_dy, dv_dx, dv_dy = _jacobian(field)
        curl = dv_dx - du_dy
        div  = du_dx + dv_dy
        return torch.cat([curl, div], dim=1)   # (B, 2, H, W)

    return F.mse_loss(_features(pred) * ocean, _features(true) * ocean)


def spectral_loss(
    pred:  torch.Tensor,
    true:  torch.Tensor,
    ocean: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between the FFT power spectra of pred and true.
    Land pixels are zeroed before the FFT so they contribute no energy.

    Args:
        pred, true: (B, 2, H, W) vector fields
        ocean:      (1, 1, H, W) float mask
    """
    def _features(field):
        masked = field * ocean
        Su = torch.fft.rfft2(masked[:, 0]).abs()   # (B, H, W//2+1)
        Sv = torch.fft.rfft2(masked[:, 1]).abs()
        return torch.stack([Su, Sv], dim=1)         # (B, 2, H, W//2+1)

    return F.mse_loss(_features(pred), _features(true))


def okubo_weiss_loss(
    pred:  torch.Tensor,
    true:  torch.Tensor,
    ocean: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between the Okubo-Weiss parameter W of pred and true, masked to ocean.

        sₙ = du/dx − dv/dy   (normal strain)
        s_s = du/dy + dv/dx  (shear strain)
        ω   = dv/dx − du/dy  (vorticity)
        W   = sₙ² + s_s² − ω²

    W < 0 → rotation-dominated (eddy cores)
    W > 0 → strain-dominated  (eddy boundaries)

    Args:
        pred, true: (B, 2, H, W) vector fields
        ocean:      (1, 1, H, W) float mask
    """
    def _ow(field):
        du_dx, du_dy, dv_dx, dv_dy = _jacobian(field)
        sn = du_dx - dv_dy   # normal strain
        ss = du_dy + dv_dx   # shear strain
        w  = dv_dx - du_dy   # vorticity
        return sn**2 + ss**2 - w**2   # (B, 1, H, W)

    return F.mse_loss(_ow(pred) * ocean, _ow(true) * ocean)


def _vorticity_cloud(
    field: torch.Tensor,
    ocean: torch.Tensor,
):
    """
    Convert the vorticity field into a weighted 2-D point cloud for geomloss.

    Returns:
        coords:  (B, N, 2) — normalised (row, col) coordinates in [0, 1]
        weights: (B, N, 1) — |ω| weights normalised to sum 1 per sample
    """
    du_dx, du_dy, dv_dx, _ = _jacobian(field)
    curl = (dv_dx - du_dy) * ocean   # (B, 1, H, W)

    B, _, H, W = curl.shape

    rows = torch.linspace(0, 1, H, device=field.device)
    cols = torch.linspace(0, 1, W, device=field.device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")   # (H, W)
    coords = torch.stack([grid_r, grid_c], dim=-1)               # (H, W, 2)
    coords = coords.view(1, H * W, 2).expand(B, -1, -1)          # (B, N, 2)

    weights = curl.abs().view(B, H * W)                          # (B, N)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    weights = weights.unsqueeze(-1)                              # (B, N, 1)

    return coords, weights


def wasserstein_loss(
    pred:        torch.Tensor,
    true:        torch.Tensor,
    ocean:       torch.Tensor,
    sinkhorn_fn,
) -> torch.Tensor:
    """
    Sinkhorn–Wasserstein distance between the vorticity point clouds of
    pred and true, averaged over the batch.

    Args:
        pred, true:  (B, 2, H, W) vector fields
        ocean:       (1, 1, H, W) float mask
        sinkhorn_fn: a geomloss.SamplesLoss instance
    """
    coords_pred, w_pred = _vorticity_cloud(pred, ocean)
    coords_true, w_true = _vorticity_cloud(true, ocean)
    dist = sinkhorn_fn(
        w_pred.squeeze(-1), coords_pred,
        w_true.squeeze(-1), coords_true,
    )
    return dist.mean()
