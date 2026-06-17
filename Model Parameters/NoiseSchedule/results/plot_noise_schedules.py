"""
Plot beta schedules and cumulative alpha_bar for all 5 noise schedules,
aligned with the definitions in Ho et al. (2020) DDPM paper.

Forward process (Eq. 2-4):
    q(x_t | x_{t-1}) = N(sqrt(1-beta_t)*x_{t-1}, beta_t*I)
    q(x_t | x_0)     = N(sqrt(alpha_bar_t)*x_0,  (1-alpha_bar_t)*I)
  where:
    alpha_t     = 1 - beta_t
    alpha_bar_t = prod_{s=1}^{t} alpha_s    (alpha_bar_0 = 1 by definition)

Usage (run from workspace root):
    py NoiseSchedule/plot_noise_schedules.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from diffusion import DDPM

SCHEDULES = ["linear", "cosine_s0001", "cosine", "cosine_s02", "cosine_s10", "quadratic", "sigmoid", "geometric"]
COLORS = {
    "linear":       "#d62728",
    "cosine_s0001": "#000080",   # dark navy  (most extreme)
    "cosine":       "#1f77b4",   # blue       (s=0.008 default)
    "cosine_s02":   "#17becf",   # cyan       (s=0.02)
    "cosine_s10":   "#aec7e8",   # light blue (s=0.10, mildest)
    "quadratic":    "#2ca02c",
    "sigmoid":      "#ff7f0e",
    "geometric":    "#9467bd",
}
T = 1000

# Paper uses t = 1..T for beta_t, and t = 0..T for alpha_bar
# (alpha_bar_0 = 1 = clean data; alpha_bar_T ≈ 0 = pure noise)
t_beta    = np.arange(1, T + 1)   # 1..1000  (x-axis for beta / per-step panels)
t_ab      = np.arange(0, T + 1)   # 0..1000  (x-axis for alpha_bar panels)
t_denoise = np.arange(1, T + 1)   # denoising step index 1..T

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for s in SCHEDULES:
    d = DDPM(T=T, beta_schedule=s)

    betas     = d.betas.numpy()        # beta_1 .. beta_T  (length T)
    alpha_bar = d.alpha_bar.numpy()    # alpha_bar_1 .. alpha_bar_T  (length T)

    # Prepend alpha_bar_0 = 1.0 so curves start at (t=0, 1.0) — clean data
    ab = np.concatenate([[1.0], alpha_bar])   # length T+1

    # Per-step noise level change = ᾱ_{t-1} - ᾱ_t  (always >= 0, length T)
    # Forward reading: noise added at each corruption step t=1..T
    # Reversed:        noise removed at each denoising step
    delta_noise = ab[:-1] - ab[1:]

    col = COLORS[s]
    lw  = 1.8
    COSINE_VARIANTS = ("cosine_s0001", "cosine", "cosine_s02", "cosine_s10")
    if s not in COSINE_VARIANTS:
        axes[0].plot(t_beta, betas,       color=col, label=s, linewidth=lw)
    else:
        axes[1].plot(t_beta, betas,       color=col, label=s, linewidth=lw)
    axes[2].plot(t_beta, delta_noise,     color=col, label=s, linewidth=lw)
    # Cumulative noise remaining during denoising:
    # index 0 = before any denoising (pure noise, 1-ab[T] ≈ 1.0)
    # index T = after all steps (clean, 1-ab[0] = 0.0)
    noise_remaining = (1.0 - ab)[::-1]   # length T+1
    axes[3].plot(t_ab,   noise_remaining, color=col, label=s, linewidth=lw)

# --- Panel 0: beta_t all schedules except cosine ---
axes[0].set_title(r"$\beta_t$  — all schedules except cosine  (shared scale)", fontsize=12)
axes[0].set_xlabel(r"Timestep $t$")
axes[0].set_ylabel(r"$\beta_t$")
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- Panel 1: beta_t cosine variants ---
axes[1].set_title(r"$\beta_t$  — cosine variants  (s=0.0001, 0.008, 0.02, 0.10)", fontsize=12)
axes[1].set_xlabel(r"Timestep $t$")
axes[1].set_ylabel(r"$\beta_t$  (cosine)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# --- Panel 2: noise added per forward step ---
axes[2].set_title(
    "Noise added per forward step  (corruption direction)\n"
    r"$\bar{\alpha}_{t-1} - \bar{\alpha}_t$ — how much noise level jumps at step $t$",
    fontsize=12
)
axes[2].set_xlabel(r"Timestep $t$  →  corruption increases right")
axes[2].set_ylabel("noise level increase per step")
axes[2].legend()
axes[2].grid(alpha=0.3)

# --- Panel 3: cumulative noise remaining during denoising ---
axes[3].set_title(
    "Noise remaining during denoising  (inference direction)\n"
    "starts at 100% pure noise, falls to 0% as clean image emerges",
    fontsize=12
)
axes[3].set_xlabel("Denoising steps completed  (0 = pure noise,  1000 = clean)")
axes[3].set_ylabel(r"noise level remaining  $(1-\bar{\alpha}_t)$")
axes[3].set_ylim(-0.05, 1.05)
axes[3].legend()
axes[3].grid(alpha=0.3)

plt.suptitle(
    r"DDPM Noise Schedule Comparison  (Ho et al. 2020,  $T=1000$,"
    r"  $\beta_1=10^{-4}$,  $\beta_T=0.02$)",
    fontsize=13
)
plt.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_schedule_comparison.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
