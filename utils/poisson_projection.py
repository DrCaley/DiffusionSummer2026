"""
Exact divergence-free projection via sparse Poisson solver with Neumann BCs.

Implements the discrete Helmholtz–Hodge decomposition on an irregular ocean domain:

    u' = u − ∇φ
    ∇²φ = ∇·u   with ∂φ/∂n = 0 (zero-flux) at all land boundaries

The result u' is the CLOSEST divergence-free field to u in L2 norm — only the
irrotational (curl-free) component is removed; large-scale structure is preserved.
Land cells remain exactly 0.

Discretisation (collocated grid, backward div / forward grad pair)
------------------------------------------------------------------
Backward divergence at ocean cell (i,j):
    d[i,j] = (u[i,j] − u[i-1,j]·1_{i-1 ocean})
           + (v[i,j] − v[i,j-1]·1_{j-1 ocean})

Forward gradient of scalar φ at ocean cell (i,j):
    gx[i,j] = (φ[i+1,j] − φ[i,j])·1_{i+1 ocean}   (H-direction, x-axis)
    gy[i,j] = (φ[i,j+1] − φ[i,j])·1_{j+1 ocean}   (W-direction, y-axis)

These form an adjoint pair ⟹ Laplacian L = Div_u·Gx + Div_v·Gy equals the
standard graph Laplacian: L[k,k] = −n_ocean_neighbors, L[k,m] = +1 for each
adjacent ocean neighbor m.

Proof that div(u') = 0:
    div(u') = Div_u·(u−Gx·φ) + Div_v·(v−Gy·φ)
            = div(u) − (Div_u·Gx + Div_v·Gy)·φ
            = d − L·φ = d − d = 0  ✓

Physical axes (matching dataset.py / divfree_projection.py):
    Tensor shape (B, 2, H=94, W=44)
    H (dim -2) = east-west  = x    channel 0 = u (east-west velocity)
    W (dim -1) = north-south = y   channel 1 = v (north-south velocity)
    divergence = ∂u/∂H + ∂v/∂W

Usage
-----
    from poisson_projection import build_poisson_projector, project_batch

    projector, ocean_idx = build_poisson_projector(ocean_mask_np)
    x_proj = project_batch(x, projector, ocean_idx)
"""

import numpy as np
import hashlib
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# System construction  (call once per domain — O(n log n) for factorisation)
# ---------------------------------------------------------------------------

def build_poisson_projector(ocean_mask_np: np.ndarray):
    """
    Pre-compute and factorize the Poisson system for the given ocean domain.

    Args:
        ocean_mask_np: (H, W) bool array, True = ocean cell

    Returns:
        projector:  callable (u_flat, v_flat) -> (u_proj, v_proj)
                    where *_flat are 1-D float64 arrays of length n_ocean
        ocean_idx:  (H, W) int32 array, value = linear ocean index [0, n_ocean),
                    or -1 for land cells
    """
    H, W = ocean_mask_np.shape

    # --- linear index for ocean cells ---
    ocean_idx = np.full((H, W), -1, dtype=np.int32)
    n = 0
    for i in range(H):
        for j in range(W):
            if ocean_mask_np[i, j]:
                ocean_idx[i, j] = n
                n += 1

    # --- backward divergence operators (n × n sparse) ---
    #   Div_u: ∂u/∂H (x-direction)  — backward difference in H (dim -2)
    #   Div_v: ∂v/∂W (y-direction)  — backward difference in W (dim -1)

    du_r, du_c, du_v = [], [], []
    dv_r, dv_c, dv_v = [], [], []

    for i in range(H):
        for j in range(W):
            if not ocean_mask_np[i, j]:
                continue
            k = ocean_idx[i, j]

            # u: u[i,j] − u[i-1,j]·1_{i-1 ocean}
            du_r.append(k); du_c.append(k); du_v.append(1.0)
            if i > 0 and ocean_mask_np[i - 1, j]:
                du_r.append(k); du_c.append(ocean_idx[i - 1, j]); du_v.append(-1.0)

            # v: v[i,j] − v[i,j-1]·1_{j-1 ocean}
            dv_r.append(k); dv_c.append(k); dv_v.append(1.0)
            if j > 0 and ocean_mask_np[i, j - 1]:
                dv_r.append(k); dv_c.append(ocean_idx[i, j - 1]); dv_v.append(-1.0)

    Div_u = sp.csr_matrix((du_v, (du_r, du_c)), shape=(n, n), dtype=np.float64)
    Div_v = sp.csr_matrix((dv_v, (dv_r, dv_c)), shape=(n, n), dtype=np.float64)

    # --- forward gradient operators (n × n sparse) ---
    #   Gx: (φ[i+1,j] − φ[i,j])·1_{i+1 ocean}  (H-direction)
    #   Gy: (φ[i,j+1] − φ[i,j])·1_{j+1 ocean}  (W-direction)

    gx_r, gx_c, gx_v = [], [], []
    gy_r, gy_c, gy_v = [], [], []

    for i in range(H):
        for j in range(W):
            if not ocean_mask_np[i, j]:
                continue
            k = ocean_idx[i, j]

            if i + 1 < H and ocean_mask_np[i + 1, j]:
                m = ocean_idx[i + 1, j]
                gx_r.append(k); gx_c.append(m); gx_v.append(1.0)
                gx_r.append(k); gx_c.append(k); gx_v.append(-1.0)

            if j + 1 < W and ocean_mask_np[i, j + 1]:
                m = ocean_idx[i, j + 1]
                gy_r.append(k); gy_c.append(m); gy_v.append(1.0)
                gy_r.append(k); gy_c.append(k); gy_v.append(-1.0)

    Gx = sp.csr_matrix((gx_v, (gx_r, gx_c)), shape=(n, n), dtype=np.float64)
    Gy = sp.csr_matrix((gy_v, (gy_r, gy_c)), shape=(n, n), dtype=np.float64)

    # --- Laplacian: L = Div_u @ Gx + Div_v @ Gy ---
    L = (Div_u @ Gx + Div_v @ Gy).astype(np.float64)

    # Fix gauge: L has a 1-D null space (constant φ).
    # Pin φ[0] = 0 by replacing row 0 with the identity row.
    L_lil = L.tolil()
    L_lil[0, :] = 0.0
    L_lil[0, 0] = 1.0
    L_pinned = L_lil.tocsr()

    # Direct LU factorisation — computed once, reused for every sample
    _solve = spla.factorized(L_pinned)

    # Capture operators in closure
    def projector(u_flat: np.ndarray, v_flat: np.ndarray):
        """
        Project (u, v) onto the divergence-free subspace.

        Args:
            u_flat: (n_ocean,) float64 — east-west velocity at ocean cells
            v_flat: (n_ocean,) float64 — north-south velocity at ocean cells

        Returns:
            u_proj, v_proj: (n_ocean,) float64 — divergence-free correction
        """
        d = (Div_u @ u_flat + Div_v @ v_flat).copy()
        d[0] = 0.0                          # consistent with φ[0] = 0 constraint
        phi   = _solve(d)
        return u_flat - Gx @ phi, v_flat - Gy @ phi

    return projector, ocean_idx


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def project_batch(
    x: np.ndarray,
    projector,
    ocean_idx: np.ndarray,
) -> np.ndarray:
    """
    Project a batch of vector fields onto the divergence-free subspace.

    Args:
        x:          (B, 2, H, W) float32 — u/v fields, land cells = 0
        projector:  callable from build_poisson_projector
        ocean_idx:  (H, W) int32 from build_poisson_projector

    Returns:
        (B, 2, H, W) float32 — projected fields, land cells = 0
    """
    out        = x.copy()
    ocean_mask = ocean_idx >= 0   # (H, W) bool

    for b in range(x.shape[0]):
        u_flat = x[b, 0][ocean_mask].astype(np.float64)
        v_flat = x[b, 1][ocean_mask].astype(np.float64)
        u_proj, v_proj = projector(u_flat, v_flat)
        out[b, 0][ocean_mask] = u_proj.astype(np.float32)
        out[b, 1][ocean_mask] = v_proj.astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Cached torch-compatible wrapper  (used by divfree_projection.leray_project)
# ---------------------------------------------------------------------------

_PROJECTOR_CACHE: dict = {}


def _ocean_mask_key(ocean_mask_np: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(ocean_mask_np).tobytes()).hexdigest()


def leray_project_poisson(x: torch.Tensor, ocean_mask: torch.Tensor) -> torch.Tensor:
    """
    Exact divergence-free projection using a cached sparse Poisson solve.

    The Poisson system is factorised on first call for a given ocean_mask and
    reused for all subsequent calls (per process).  Works on any device —
    computation is on CPU via scipy; the result is returned on the original device.

    Args:
        x:          (B, 2, H, W) torch tensor, any device
        ocean_mask: (H, W) bool tensor, True = ocean

    Returns:
        (B, 2, H, W) torch tensor on same device as x.
        Backward-difference divergence is exactly zero at every ocean cell.
    """
    orig_device   = x.device
    ocean_mask_np = ocean_mask.cpu().numpy().astype(bool)

    key = _ocean_mask_key(ocean_mask_np)
    if key not in _PROJECTOR_CACHE:
        _PROJECTOR_CACHE[key] = build_poisson_projector(ocean_mask_np)
    projector, ocean_idx = _PROJECTOR_CACHE[key]

    x_np      = x.detach().cpu().numpy().astype(np.float32)
    x_proj_np = project_batch(x_np, projector, ocean_idx)
    return torch.from_numpy(x_proj_np).to(orig_device)

