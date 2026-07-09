"""
Bar graph for ensemble n_samples experiment results.
Reads ensemble_nsample_results.txt and produces a two-panel figure:
  Left:  mean RMSE ± std per ensemble size
  Right: mean inference time ± std per ensemble size

Run from this folder:
    python plot_nsample_results.py
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Parse the results file
# ---------------------------------------------------------------------------

results_file = os.path.join(os.path.dirname(__file__), "ensemble_nsample_results.txt")

n_samples_list  = []
mean_rmse_list  = []
std_rmse_list   = []
mean_time_list  = []
std_time_list   = []

# Match data rows in the summary table (integers then floats)
row_pattern = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)

with open(results_file) as f:
    for line in f:
        m = row_pattern.match(line)
        if m:
            n_samples_list.append(int(m.group(1)))
            mean_time_list.append(float(m.group(2)))
            std_time_list.append(float(m.group(3)))
            mean_rmse_list.append(float(m.group(4)))
            std_rmse_list.append(float(m.group(5)))

n   = np.array(n_samples_list)
mr  = np.array(mean_rmse_list)
sr  = np.array(std_rmse_list)
mt  = np.array(mean_time_list)
st  = np.array(std_time_list)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

x      = np.arange(len(n))
width  = 0.55
color  = "#4C8FBF"
ecolor = "#1a1a2e"

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(
    "Ensemble n_samples: RMSE and Inference Time\n"
    "T=1000, stride=10, resample=1, 50 seeds",
    fontsize=13, fontweight="bold",
)

# ---- Left: RMSE ----
ax = axes[0]
bars = ax.bar(x, mr, width, yerr=sr, capsize=5,
              color=color, edgecolor="white", error_kw={"ecolor": ecolor, "lw": 1.5})
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in n], fontsize=11)
ax.set_xlabel("n_samples (ensemble size)", fontsize=11)
ax.set_ylabel("Mean RMSE (±1 std, 50 seeds)", fontsize=11)
ax.set_title("RMSE vs Ensemble Size", fontsize=12)
ax.set_ylim(0, max(mr + sr) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for bar, val, s in zip(bars, mr, sr):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + s + 0.001,
        f"{val:.4f}",
        ha="center", va="bottom", fontsize=9, color="black",
    )

# ---- Right: Time ----
ax = axes[1]
bars = ax.bar(x, mt, width, yerr=st, capsize=5,
              color="#E07B54", edgecolor="white", error_kw={"ecolor": ecolor, "lw": 1.5})
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in n], fontsize=11)
ax.set_xlabel("n_samples (ensemble size)", fontsize=11)
ax.set_ylabel("Mean inference time (s, ±1 std)", fontsize=11)
ax.set_title("Inference Time vs Ensemble Size", fontsize=12)
ax.set_ylim(0, max(mt + st) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for bar, val, s in zip(bars, mt, st):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + s + 0.2,
        f"{val:.1f}s",
        ha="center", va="bottom", fontsize=9, color="black",
    )

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "nsample_bar_graph.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
