"""
Posterior-sampling procedures for the CONDITIONAL stream-function DDPM.

All three procedures draw an *ensemble* of N divergence-free fields from the
SAME fixed conditioning (sparse robot observations + temporal priors + geometry)
trained by `train_cond.py` (pred_type = "x0_streamfn_cond").  They differ only
in HOW they use the observations during the reverse diffusion:

  1. vanilla_ensemble  — observations enter ONLY as soft input channels.
       Diversity comes purely from the diffusion noise.  This is the project
       baseline: "run the model N times."  Fast, maximally diverse, but the
       observations are a gentle hint the network may partly ignore.

  2. particle_filter    — Sequential-Monte-Carlo ("particle diffusion").
       N particles are denoised together; at each step each particle is
       re-weighted by how well its predicted clean field matches the observed
       pixels, and particles are resampled when they degenerate (ESS gate).
       This concentrates the ensemble on observation-consistent flows while
       leaving the unobserved far field free to diverge.

  3. dps_ensemble       — Diffusion Posterior Sampling (gradient guidance).
       Each sample is nudged at every step along -∇_xt ||M⊙(x̂₀ - y)||², i.e.
       toward agreement with the observations.  No resampling, so diversity is
       preserved continuously; tends to give the best accuracy-vs-diversity
       trade-off where the data informs the answer.

Every procedure returns ``members`` — a list of N numpy (2, H, W) fields, each
divergence-free by construction (curl of the network's scalar stream function).

The observations used for weighting/guidance are read straight from the
conditioning tensor (the SINGLE source of truth, identical to training):
    cond[0:2] = obs_u, obs_v  (true field on the path, 0 elsewhere)
    cond[2:3] = path_mask     (1 on observed cells)
so the likelihood is exactly the soft constraint the model was conditioned on.
"""

import os
import sys

import numpy as np
import torch

# --- path shim: works from workspace root or the flat server layout ---------
_here  = os.path.dirname(os.path.abspath(__file__))
_root  = os.path.normpath(os.path.join(_here, "..", ".."))
for _p in [_root, os.path.join(_root, "utils"),
           os.path.join(_root, "DDPM", "model"), os.path.join(_here, "..", "model")]:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from diffusion import eps_wrapper_for, x0_from_output   # noqa: E402


# ===========================================================================
# Shared helpers
# ===========================================================================

def _x0_hat(stream_model, diffusion, xt, t_t, cond, pred_type):
    """Model forward → divergence-free x̂₀, honouring the parameterization.

    For x0-prediction the network output IS x̂₀; for v-prediction it is v̂ and
    x̂₀ = √ᾱ·x_t − √(1−ᾱ)·v̂ (still divergence-free).
    """
    out = stream_model(xt, t_t, cond)
    return x0_from_output(diffusion, xt, out, t_t, pred_type)

def _obs_from_cond(cond, ocean, device):
    """Pull the observation target y and the observed-cell mask m from cond.

    Returns y (1, 2, H, W) and m (1, 1, H, W), both on ``device`` and already
    restricted to ocean cells.
    """
    y = cond[0:2].to(device)[None]                  # (1, 2, H, W)
    m = cond[2:3].to(device)[None] * ocean          # (1, 1, H, W) observed ∩ ocean
    return y, m


def _init_latent(diffusion, n, H, W, ocean, device, seed):
    """Sample the initial x_T for n particles (masked to ocean, scaled)."""
    torch.manual_seed(seed)
    x = diffusion._sample_noise(torch.empty(n, 2, H, W, device=device))
    return x * diffusion.noise_scale * ocean


def _posterior_step_from_x0(diffusion, xt, x0_hat, t_int, t_prev_int, noise=None):
    """One DDPM reverse step given an ALREADY-computed x̂₀ (no extra model call).

    Mirrors `DDPM.p_sample_step` exactly, but takes x0_hat directly so the
    guided samplers can reuse a single forward pass.  Returns x_{t_prev}.
    """
    ns = diffusion.noise_scale
    x0 = x0_hat.clamp(-3.0 * ns, 3.0 * ns)
    ab = diffusion.alpha_bar[t_int]
    if t_prev_int < 0:
        return x0
    ab_prev  = diffusion.alpha_bar[t_prev_int]
    beta_eff = 1.0 - ab / ab_prev
    var      = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
    coef1    = ab_prev.sqrt() * beta_eff / (1.0 - ab)
    coef2    = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
    mean     = coef1 * x0 + coef2 * xt
    if noise is None:
        noise = diffusion._sample_noise(xt)
    return mean + var.sqrt() * ns * noise


def _systematic_resample(w, device):
    """Low-variance systematic resampling.  w: (N,) normalized.  Returns idx (N,)."""
    n = w.shape[0]
    u0 = torch.rand((), device=device) / n
    positions = u0 + torch.arange(n, device=device) / n
    cumsum = torch.cumsum(w, dim=0)
    cumsum[-1] = 1.0
    idx = torch.searchsorted(cumsum, positions)
    return idx.clamp(max=n - 1)


def _members_to_numpy(xt, ocean):
    xt = xt * ocean
    return [xt[i].detach().cpu().numpy().astype(np.float32) for i in range(xt.shape[0])]


# ===========================================================================
# 1.  Vanilla ensemble — observations as soft input channels only
# ===========================================================================

@torch.no_grad()
def vanilla_ensemble(stream_model, diffusion, cond, land_np, *,
                     n_members=8, inference_steps=100, device="cpu", seed=0,
                     pred_type="x0_streamfn_cond"):
    """Run the conditional model N times; diversity from the diffusion noise."""
    H, W   = land_np.shape
    ocean  = torch.from_numpy(~land_np).float().to(device)[None, None]
    cond_b = cond.unsqueeze(0).to(device)                      # (1, C, H, W)
    eps    = eps_wrapper_for(stream_model, diffusion, pred_type,
                             cond=cond_b).to(device)

    xt = _init_latent(diffusion, n_members, H, W, ocean, device, seed)
    for t_int, t_prev_int in diffusion.build_inference_schedule(inference_steps):
        xt = diffusion.p_sample_step(eps, xt, t_int, t_prev_int) * ocean
    return _members_to_numpy(xt, ocean), {}


# ===========================================================================
# 2.  Particle filter (SMC) — reweight + resample by observation likelihood
# ===========================================================================

@torch.no_grad()
def particle_filter(stream_model, diffusion, cond, land_np, *,
                    n_members=8, inference_steps=100, device="cpu", seed=0,
                    obs_sigma=0.1, ess_frac=0.5,
                    pred_type="x0_streamfn_cond"):
    """Sequential-Monte-Carlo "particle diffusion".

    N particles are denoised jointly.  At each step the particle weight is
    updated by the *change* in observation log-likelihood of its predicted
    clean field (telescoping Feynman–Kac potential), and particles are
    resampled (systematic) whenever the effective sample size drops below
    ``ess_frac · N``.  ``obs_sigma`` is the assumed observation noise (in the
    normalized field units): smaller => trust the observations harder.
    """
    H, W   = land_np.shape
    ocean  = torch.from_numpy(~land_np).float().to(device)[None, None]
    cond_b = cond.unsqueeze(0).to(device)
    cond_n = cond_b.expand(n_members, -1, -1, -1)
    y, m   = _obs_from_cond(cond, ocean, device)               # (1,2,H,W),(1,1,H,W)
    inv2s2 = 1.0 / (2.0 * obs_sigma ** 2)

    xt   = _init_latent(diffusion, n_members, H, W, ocean, device, seed)
    logw = torch.zeros(n_members, device=device)
    ll_prev = None
    resamples = 0

    schedule = diffusion.build_inference_schedule(inference_steps)
    for t_int, t_prev_int in schedule:
        t_t = torch.full((n_members,), t_int, device=device, dtype=torch.long)
        x0_hat = _x0_hat(stream_model, diffusion, xt, t_t, cond_n, pred_type)

        # Observation log-likelihood of each particle's clean-field estimate.
        resid = (x0_hat - y) * m
        ll    = -(resid ** 2).sum(dim=(1, 2, 3)) * inv2s2      # (N,)

        # Telescoping incremental weight (FK potential difference).
        if ll_prev is not None:
            logw = logw + (ll - ll_prev)
        w = torch.softmax(logw, dim=0)

        # Resample when particles degenerate — but NEVER on the final step:
        # the last reverse step is deterministic (no noise), so resampling there
        # would leave the copies identical and collapse the ensemble to one field.
        ess = 1.0 / (w ** 2).sum().clamp(min=1e-12)
        if ess < ess_frac * n_members and t_prev_int >= 0:
            idx     = _systematic_resample(w, device)
            xt      = xt[idx]
            x0_hat  = x0_hat[idx]
            ll      = ll[idx]
            logw    = torch.zeros(n_members, device=device)
            resamples += 1
        ll_prev = ll

        xt = _posterior_step_from_x0(diffusion, xt, x0_hat, t_int, t_prev_int) * ocean

    return _members_to_numpy(xt, ocean), {"resamples": resamples}


# ===========================================================================
# 3.  Diffusion Posterior Sampling (DPS) — gradient guidance toward observations
# ===========================================================================

def dps_ensemble(stream_model, diffusion, cond, land_np, *,
                 n_members=8, inference_steps=100, device="cpu", seed=0,
                 obs_sigma=0.1, zeta=0.05,
                 pred_type="x0_streamfn_cond"):
    """Gradient-guided ensemble.

    Each sample follows the standard reverse process but is additionally nudged
    each step along the observation-matching gradient
        -∇_xt ‖M⊙(x̂₀(xt) - y)‖²,
    normalized per-particle and scaled by ``zeta`` (effective step in field
    units).  ``obs_sigma`` only rescales the loss and is folded into ``zeta``.
    No resampling, so the N samples stay continuously diverse.
    """
    H, W   = land_np.shape
    ocean  = torch.from_numpy(~land_np).float().to(device)[None, None]
    cond_b = cond.unsqueeze(0).to(device)
    cond_n = cond_b.expand(n_members, -1, -1, -1)
    y, m   = _obs_from_cond(cond, ocean, device)

    xt = _init_latent(diffusion, n_members, H, W, ocean, device, seed)
    for t_int, t_prev_int in diffusion.build_inference_schedule(inference_steps):
        xt = xt.detach().requires_grad_(True)
        t_t = torch.full((n_members,), t_int, device=device, dtype=torch.long)
        x0_hat = _x0_hat(stream_model, diffusion, xt, t_t, cond_n, pred_type)

        resid = (x0_hat - y) * m
        loss  = (resid ** 2).sum() / (2.0 * obs_sigma ** 2)    # particles independent
        grad  = torch.autograd.grad(loss, xt)[0]               # (N, 2, H, W)

        with torch.no_grad():
            x_prev = _posterior_step_from_x0(
                diffusion, xt.detach(), x0_hat.detach(), t_int, t_prev_int)
            # Per-particle unit-normalized guidance step (stable across t).
            gnorm  = grad.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1, 1)
            x_prev = x_prev - zeta * grad / gnorm
            xt = (x_prev * ocean).detach()

    return _members_to_numpy(xt, ocean), {}


# ===========================================================================
# Registry — selectable by name from the comparison driver
# ===========================================================================

SAMPLERS = {
    "vanilla":  vanilla_ensemble,
    "particle": particle_filter,
    "dps":      dps_ensemble,
}
