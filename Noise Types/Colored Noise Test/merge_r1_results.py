"""
merge_r1_results.py

Merges white_vs_annealed_r1 and white_vs_red_r1 into a single
outputs/r1_combined/ folder:
  - copies all per-sample PNGs (renamed to include noise type)
  - writes a unified summary.txt
  - writes a bar chart rmse_bars_r1.png comparing all three models
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
_OUT    = os.path.join(_HERE, "outputs")
_MERGED = os.path.join(_OUT, "r1_combined")
os.makedirs(_MERGED, exist_ok=True)

# ── Parse a summary.txt ──────────────────────────────────────────────────────

def parse_summary(path):
    """Return dict: model_label -> list of per-sample RMSE floats."""
    data = {}
    headers = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.strip().startswith("sample"):
                headers = line.split()
                continue
            if headers and line.strip():
                parts = line.split()
                if len(parts) != len(headers):
                    continue
                for col, label in enumerate(headers[1:], start=1):
                    data.setdefault(label, []).append(float(parts[col]))
    return data

combined_data = parse_summary(os.path.join(_MERGED, "summary.txt"))

models = [
    ("White",    combined_data["White"]),
    ("Red",      combined_data["Red"]),
    ("Annealed", combined_data["Annealed"]),
]

N = len(models[0][1])
print(f"Loaded {N} samples from r1_combined/summary.txt")

# ── Bar chart ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RePaint r=1 — White / Red / Annealed (20 seeds, best ckpt)", fontsize=13)

labels = [lbl for lbl, _ in models]
means  = [np.mean(v) for _, v in models]
stds   = [np.std(v)  for _, v in models]
colors = ["#4c9be8", "#c44ce8", "#e8c94c"]

# Left: mean RMSE with std error bars
ax = axes[0]
bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor="black",
              width=0.5, capsize=6)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(means) * 0.02,
            f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Mean RMSE (normalised)")
ax.set_title("Mean RMSE ± Std")
ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

# Right: per-sample scatter + mean bar
ax2 = axes[1]
x_pos = np.arange(len(models))
for xi, (label, vals) in zip(x_pos, models):
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(vals))
    ax2.scatter(np.full(len(vals), xi) + jitter, vals,
                color=colors[xi], alpha=0.6, s=30, zorder=3)
    ax2.hlines(np.mean(vals), xi - 0.25, xi + 0.25,
               color=colors[xi], linewidth=2.5, zorder=4, label=f"{label} mean")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.set_ylabel("RMSE (normalised)")
ax2.set_title("Per-sample RMSE")
ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
ax2.set_axisbelow(True)
ax2.legend(fontsize=9)

plt.tight_layout()
bar_path = os.path.join(_MERGED, "rmse_bars_r1.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {bar_path}")
print("Done.")
