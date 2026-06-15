from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _prepare_axes(ax: plt.Axes, title: str, land_mask: np.ndarray | torch.Tensor | None) -> None:
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    if land_mask is not None:
        mask = land_mask.detach().cpu().numpy() if isinstance(land_mask, torch.Tensor) else land_mask
        ax.imshow(mask, cmap="Greys", alpha=0.25, origin="upper")


def _quiver(ax: plt.Axes, field: torch.Tensor | np.ndarray, step: int = 4) -> None:
    array = field.detach().cpu().numpy() if isinstance(field, torch.Tensor) else field
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    u = array[0]
    v = array[1]
    height, width = u.shape
    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ax.quiver(grid_x, grid_y, u[::step, ::step], v[::step, ::step], color="white", scale=18, width=0.003)


def plot_actual_field(field: torch.Tensor | np.ndarray, land_mask: np.ndarray | torch.Tensor | None, save_path: str | Path, title: str = "Actual field") -> None:
    array = field.detach().cpu().numpy() if isinstance(field, torch.Tensor) else field
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude = np.sqrt(array[0] ** 2 + array[1] ** 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(magnitude, origin="upper", cmap="viridis")
    _prepare_axes(ax, title, land_mask)
    _quiver(ax, array)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_prediction_field(field: torch.Tensor | np.ndarray, land_mask: np.ndarray | torch.Tensor | None, observation_mask: np.ndarray | torch.Tensor | None, save_path: str | Path, title: str = "Predicted field") -> None:
    array = field.detach().cpu().numpy() if isinstance(field, torch.Tensor) else field
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude = np.sqrt(array[0] ** 2 + array[1] ** 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(magnitude, origin="upper", cmap="magma")
    _prepare_axes(ax, title, land_mask)
    _quiver(ax, array)
    if observation_mask is not None:
        mask = observation_mask.detach().cpu().numpy() if isinstance(observation_mask, torch.Tensor) else observation_mask
        points = np.argwhere(mask)
        if len(points) > 0:
            ax.scatter(points[:, 1], points[:, 0], s=6, c="cyan", alpha=0.8, label="Robot path")
            ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss_field(field: torch.Tensor | np.ndarray, land_mask: np.ndarray | torch.Tensor | None, save_path: str | Path, title: str = "Absolute error") -> None:
    array = field.detach().cpu().numpy() if isinstance(field, torch.Tensor) else field
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim == 3:
        array = np.sqrt(array[0] ** 2 + array[1] ** 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(array, origin="upper", cmap="inferno")
    _prepare_axes(ax, title, land_mask)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
