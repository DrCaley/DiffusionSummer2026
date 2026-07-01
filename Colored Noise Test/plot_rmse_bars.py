"""
plot_rmse_bars.py — Bar graphs of mean RMSE for batch_best and batch_epoch100.

Reads summary.txt from each output directory, then saves:
    outputs/rmse_bars_best.png
    outputs/rmse_bars_epoch100.png
    outputs/rmse_bars_combined.png   (side-by-side comparison)

Usage (from workspace root):
    python "Colored Noise Test/plot_rmse_bars.py"
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
BEST_SUM = os.path.join(_HERE, "outputs", "batch_best",     "summary.txt")
E100_SUM = os.path.join(_HERE, "outputs", "batch_epoch100", "summary.txt")
OUT_DIR  = os.path.join(_HERE, "outputs")

# ── Parser ───────────────────────────────────────────────────────────────────

def parse_summary(path: str) -> dict:
    """
    Returns dict with keys:
        'ckpt'    : str
        'models'  : list of (label, mean, std, min, max)
    """
    with open(path) as f:
        text = f.read()

    # Checkpoint type line
    m = re.search(r"Checkpoint type\s*:\s*(\S+)", text)
    ckpt = m.group(1) if m else "unknown"

    # Table rows: after the dashed separator
    # Format: Label  epoch  val_loss  Mean RMSE  Std  Min  Max
    rows = []
    in_table = False
    for line in text.splitlines():
        if re.match(r"-{20,}", line):
            in_table = True
            continue
        if in_table:
            if not line.strip() or line.startswith("Per-"):
                break
            parts = line.split()
            # Label may be two words (e.g. "Pink (full)")
            # Find the column with epoch (pure integer)
            # Layout: label... epoch val_loss mean std min max
            # epoch is always the first purely-numeric token after the label
            # Safer: split from right — last 5 tokens are val_loss mean std min max
            # and the token before that is epoch
            tokens = line.split()
            label_end = len(tokens) - 6   # 6 numeric columns at the end
            label  = " ".join(tokens[:label_end])
            mean   = float(tokens[-4])
            std    = float(tokens[-3])
            lo     = float(tokens[-2])
            hi     = float(tokens[-1])
            rows.append((label, mean, std, lo, hi))

    return {"ckpt": ckpt, "models": rows}


# ── Plotter ──────────────────────────────────────────────────────────────────

COLORS = [
    "#4C72B0",  # White
    "#DD8452",  # Pink
    "#55A868",  # Red
    "#C44E52",  # Pink (full)
    "#8172B3",  # Red (full)
    "#937860",  # Annealed
]


def bar_graph(data: dict, out_path: str, title: str):
    models = data["models"]
    labels = [m[0] for m in models]
    means  = np.array([m[1] for m in models])
    stds   = np.array([m[2] for m in models])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=COLORS[:len(labels)], edgecolor="black", linewidth=0.7,
                  error_kw=dict(elinewidth=1.5, ecolor="black"))

    # Value labels on top of each bar
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Mean RMSE (10 samples ± 1 std)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means + stds) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def combined_graph(best_data: dict, e100_data: dict, out_path: str):
    """Side-by-side grouped bar chart for both checkpoints on one figure."""
    models_b = best_data["models"]
    models_e = e100_data["models"]

    # Assume same model order
    labels = [m[0] for m in models_b]
    means_b = np.array([m[1] for m in models_b])
    stds_b  = np.array([m[2] for m in models_b])
    means_e = np.array([m[1] for m in models_e])
    stds_e  = np.array([m[2] for m in models_e])

    n  = len(labels)
    x  = np.arange(n)
    w  = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bars_b = ax.bar(x - w/2, means_b, w, yerr=stds_b, capsize=5,
                    color="#4C72B0", edgecolor="black", linewidth=0.7, label="Best ckpt",
                    error_kw=dict(elinewidth=1.5, ecolor="black"))
    bars_e = ax.bar(x + w/2, means_e, w, yerr=stds_e, capsize=5,
                    color="#DD8452", edgecolor="black", linewidth=0.7, label="Epoch-100 ckpt",
                    error_kw=dict(elinewidth=1.5, ecolor="black"))

    # Value labels
    for bar, mean in zip(bars_b, means_b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=7.5, color="#4C72B0")
    for bar, mean in zip(bars_e, means_e):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=7.5, color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Mean RMSE (10 samples ± 1 std)", fontsize=11)
    ax.set_title("All-model RMSE comparison: Best vs Epoch-100 checkpoints", fontsize=13, fontweight="bold")
    ymax = max(np.max(means_b + stds_b), np.max(means_e + stds_e))
    ax.set_ylim(0, ymax * 1.28)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    missing = [p for p in (BEST_SUM, E100_SUM) if not os.path.isfile(p)]
    if missing:
        print("Summary files not yet available:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    best = parse_summary(BEST_SUM)
    e100 = parse_summary(E100_SUM)

    bar_graph(
        best,
        os.path.join(OUT_DIR, "rmse_bars_best.png"),
        "RMSE by model — best checkpoint",
    )
    bar_graph(
        e100,
        os.path.join(OUT_DIR, "rmse_bars_epoch100.png"),
        "RMSE by model — epoch-100 checkpoint",
    )
    combined_graph(
        best, e100,
        os.path.join(OUT_DIR, "rmse_bars_combined.png"),
    )


if __name__ == "__main__":
    main()
