"""
Shared quiver-plot helper — used by both the DDPM and GP inpainting visualisers.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool"):
    """
    Draw a quiver (arrow) plot of a 2D vector field on *ax*.

    Args:
        ax:        matplotlib Axes
        u, v:      (H, W) east-west and north-south components
        land_mask: (H, W) bool, True = land (drawn black)
        title:     axes title string
        step:      subsampling step for the quiver grid (default 2)
        cmap:      colourmap for arrow colour (default "cool")
    """
    H, W = u.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u[::step, ::step]
    vq = v[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    mask = ~np.isnan(uq) & ~land_mask[::step, ::step]
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
        cmap=cmap, clim=(0, np.nanpercentile(mq[mask], 98) if mask.any() else 1),
        scale=12, width=0.003, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
