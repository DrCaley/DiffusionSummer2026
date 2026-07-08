"""
Generate a comparison image of White / Pink / Red noise model RMSE results
at stride=10 over 10 test samples.

Output: Colored Noise Test/outputs/noise_comparison.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Data from batch evaluation (stride=10, 10 test samples)
# ---------------------------------------------------------------------------

models = ["White", "Pink", "Red"]
colors = ["#4e79a7", "#f28e2b", "#e15759"]

per_sample = {
    "White": [0.1880, 0.0738, 0.1873, 0.1424, 0.1113, 0.0516, 0.1566, 0.0656, 0.0880, 0.1123],
    "Pink":  [0.1999, 0.1182, 0.0763, 0.1078, 0.0819, 0.1950, 0.4273, 0.2910, 0.1432, 0.5590],
    "Red":   [0.1745, 0.0938, 0.0779, 0.0700, 0.0625, 0.1815, 0.2491, 0.1726, 0.0506, 0.1151],
}

means = {m: np.mean(per_sample[m]) for m in models}
stds  = {m: np.std(per_sample[m])  for m in models}
mins  = {m: np.min(per_sample[m])  for m in models}
maxs  = {m: np.max(per_sample[m])  for m in models}

n_samples = 10
sample_ids = list(range(1, n_samples + 1))

# ---------------------------------------------------------------------------
# Figure: 1 row of 2 panels
#   Left:  bar chart of mean RMSE ± std
#   Right: per-sample RMSE line plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Colored Noise Models — RePaint Reconstruction (stride=10, r=10, 10 test samples)",
    fontsize=13, fontweight="bold"
)

# ---- Left: bar chart ----
ax = axes[0]
x = np.arange(len(models))
bars = ax.bar(
    x,
    [means[m] for m in models],
    yerr=[stds[m] for m in models],
    color=colors,
    capsize=6,
    width=0.5,
    edgecolor="black",
    linewidth=0.8,
    error_kw=dict(elinewidth=1.5, ecolor="black"),
)

# Annotate mean values
for rect, m in zip(bars, models):
    h = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        h + stds[m] + 0.005,
        f"{means[m]:.4f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

ax.set_xticks(x)
ax.set_xticklabels([f"{m}\nNoise" for m in models], fontsize=11)
ax.set_ylabel("Mean RMSE (normalised units)", fontsize=11)
ax.set_title("Mean RMSE ± Std Dev", fontsize=12)
ax.set_ylim(0, max(maxs.values()) * 1.25)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)

# ---- Right: per-sample line plot ----
ax = axes[1]
for m, c in zip(models, colors):
    ax.plot(sample_ids, per_sample[m], marker="o", color=c, linewidth=1.8,
            markersize=5, label=f"{m}  (μ={means[m]:.4f})")
    ax.fill_between(
        sample_ids,
        [means[m] - stds[m]] * n_samples,
        [means[m] + stds[m]] * n_samples,
        alpha=0.10, color=c
    )

ax.set_xlabel("Test sample index", fontsize=11)
ax.set_ylabel("RMSE (normalised units)", fontsize=11)
ax.set_title("Per-sample RMSE  (stride=10)", fontsize=12)
ax.set_xticks(sample_ids)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "noise_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\nSummary (stride=10, 10 test samples):")
print(f"{'Model':>6}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print("-" * 46)
for m in models:
    print(f"{m:>6}  {means[m]:>8.4f}  {stds[m]:>8.4f}  {mins[m]:>8.4f}  {maxs[m]:>8.4f}")
