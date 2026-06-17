"""
Divergence-free noise sampler for 2D vector fields.

How it works — Fourier-space Helmholtz projection
--------------------------------------------------
Any 2D vector field F = (u, v) can be uniquely decomposed (Helmholtz) into:

    F = F_div_free  +  F_curl_free

where:
  * F_div_free has zero divergence  (∂u/∂x + ∂v/∂y = 0)
  * F_curl_free has zero curl       (∂v/∂x − ∂u/∂y = 0)

In Fourier space this decomposition is diagonal: each wavenumber k = (kx, ky)
can be handled independently.  For a Fourier mode û(k), v̂(k) the
divergence-free projection is just the orthogonal complement of k:

    dot   = kx·û + ky·v̂          (component along k — the "curl-free" part)
    û'    = û − kx · dot / |k|²
    v̂'    = v̂ − ky · dot / |k|²

After this, kx·û' + ky·v̂' = 0 exactly.
The k=0 (DC / mean) mode has |k|²=0, so we leave it unchanged.

Implementation steps
--------------------
1.  Sample two independent Gaussian fields  (u, v) ∼ N(0, I).
2.  Apply rfft2 to get complex Fourier coefficients.
3.  Build the wavenumber grid (kx via rfftfreq, ky via fftfreq).
4.  Apply the Helmholtz projection to each Fourier mode.
5.  Apply irfft2 to get back a real spatial field.
6.  Normalise each (sample, channel) to unit std so statistics match randn.

The output is divergence-free in the periodic-domain sense — this is exact,
not approximate.  It has the same spatial extent and dtype as randn would.
"""

import torch


NOISE_TYPES = ("gaussian", "div_free")


def divergence_free_noise(
    shape:            tuple,
    device:           str                   = "cpu",
    spectral_filter:  torch.Tensor | None   = None,
) -> torch.Tensor:
    """
    Sample a batch of divergence-free Gaussian noise vectors.

    Args:
        shape:           (B, 2, H, W) — must have exactly 2 channels (u, v)
        device:          torch device string
        spectral_filter: Optional (H, W) real tensor.  When provided, the
                         Fourier coefficients of both u and v are multiplied
                         by this filter *before* Helmholtz projection, giving
                         noise whose power spectrum matches the data rather
                         than white (flat) noise.  Should be the amplitude
                         spectrum sqrt(P_data(kx, ky)) normalised to mean=1.

    Returns:
        Divergence-free noise of shape (B, 2, H, W), unit std.
    """
    B, C, H, W = shape
    assert C == 2, "divergence_free_noise requires exactly 2 channels (u, v)"

    # MPS does not support complex FFT — generate noise on CPU, move to target device at end
    orig_device = device
    cpu_device  = "cpu"

    # ------------------------------------------------------------------
    # 1. Sample isotropic Gaussian noise  (on CPU for FFT compatibility)
    # ------------------------------------------------------------------
    eps = torch.randn(B, 2, H, W, device=cpu_device)

    # ------------------------------------------------------------------
    # 2. Full 2D FFT  →  (B, 2, H, W)  complex
    #    Using fft2 (not rfft2) because the Helmholtz projection must
    #    preserve Hermitian symmetry F[-k] = conj(F[k]) for ifft2 to
    #    produce a real signal.  With fft2 all modes are stored, so
    #    the projection preserves symmetry automatically — EXCEPT at the
    #    Nyquist modes (h = H//2 when H is even, w = W//2 when W is even),
    #    which map to themselves under the conjugate operation and are not
    #    handled correctly by the projection formula.  We zero those modes
    #    out before projecting; they contribute negligible energy.
    # ------------------------------------------------------------------
    eps_f = torch.fft.fft2(eps)
    hat_u = eps_f[:, 0].clone()   # (B, H, W)
    hat_v = eps_f[:, 1].clone()   # (B, H, W)

    # Zero Nyquist modes to maintain Hermitian symmetry after projection.
    if H % 2 == 0:
        hat_u[:, H // 2, :] = 0.0
        hat_v[:, H // 2, :] = 0.0
    if W % 2 == 0:
        hat_u[:, :, W // 2] = 0.0
        hat_v[:, :, W // 2] = 0.0

    # ------------------------------------------------------------------
    # 3. Wavenumber grids (normalised cycles per grid cell)
    #
    #    Physical layout of (H, W) = (94, 44) for this dataset:
    #      dim -2 (H = 94) = east-west (x) axis  → kx = fftfreq(H)
    #      dim -1 (W = 44) = north-south (y) axis → ky = fftfreq(W)
    #
    #    Divergence-free condition:  kx·û + ky·v̂ = 0
    #      kx pairs with u (east-west velocity, channel 0)
    #      ky pairs with v (north-south velocity, channel 1)
    # ------------------------------------------------------------------
    kx = torch.fft.fftfreq(H, d=1.0, device=cpu_device).view(H, 1)   # (H, 1)
    ky = torch.fft.fftfreq(W, d=1.0, device=cpu_device).view(1, W)    # (1, W)
    k2 = kx ** 2 + ky ** 2   # (H, W)

    # ------------------------------------------------------------------
    # 4. Helmholtz projection onto the divergence-free subspace
    #
    #    dot    = kx·û + ky·v̂       (divergence in Fourier space)
    #    û'     = û  − kx·dot / |k|²
    #    v̂'     = v̂  − ky·dot / |k|²
    #
    #    Verification: kx·û' + ky·v̂'
    #      = (kx·û + ky·v̂) − (kx² + ky²)·dot / k²
    #      = dot − dot = 0  ✓
    #
    #    At k=0 (DC): kx=ky=0 so dot=0 and the correction term is 0;
    #    k2_safe replaces 0 with 1 to avoid division by zero (harmless).
    # ------------------------------------------------------------------
    k2_safe = torch.where(k2 > 0.0, k2, torch.ones_like(k2))
    # ------------------------------------------------------------------
    # 4a. Apply spectral coloring (optional)
    #
    #     Multiply both Fourier components by the amplitude filter so the
    #     noise has the same spectral shape as the training data instead of
    #     white (flat) noise.  Helmholtz projection is linear so it commutes
    #     with this scaling: coloring then projecting = projecting then coloring.
    #     We do it before the projection so the projection sees correlated
    #     modes and the result stays on the div-free subspace.
    # ------------------------------------------------------------------
    if spectral_filter is not None:
        sf     = spectral_filter.to(cpu_device)          # (H, W) real
        hat_u  = hat_u * sf
        hat_v  = hat_v * sf

    dot = kx * hat_u + ky * hat_v          # (B, H, W) complex

    hat_u_proj = hat_u - kx * dot / k2_safe
    hat_v_proj = hat_v - ky * dot / k2_safe

    # ------------------------------------------------------------------
    # 5. Inverse FFT.  After Nyquist-zeroing + Helmholtz projection,
    #    hat_u_proj and hat_v_proj are Hermitian-symmetric, so the
    #    imaginary part of ifft2 is ~0 (float32 rounding only).
    # ------------------------------------------------------------------
    u_proj = torch.fft.ifft2(hat_u_proj).real   # (B, H, W)
    v_proj = torch.fft.ifft2(hat_v_proj).real   # (B, H, W)

    eps_proj = torch.stack([u_proj, v_proj], dim=1)    # (B, 2, H, W)

    # ------------------------------------------------------------------
    # 6. Normalise to unit std per sample using a single scalar for both
    #    channels so the divergence-free property is preserved.
    #    (Normalising u and v by different stds would break ∂u'/∂x + ∂v'/∂y = 0.)
    #    std is computed over both channels and all spatial positions.
    # ------------------------------------------------------------------
    std = eps_proj.flatten(1).std(dim=1).view(B, 1, 1, 1).clamp(min=1e-8)
    return (eps_proj / std).to(orig_device)
