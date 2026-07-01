import numpy as np
import matplotlib.pyplot as plt

methods = [
    "RePaint r=10",
    "RePaint r=1",
    "DPS  ζ=0.5",
    "DPS  ζ=0.04",
    "RePaint+DPS\nζ=0.5, r=10",
    "RePaint+DPS\nr=1, ζ=0.5",
]

# T=1000 stride=10, from respective runs
rmse  = [0.1055, 0.0772, 0.1033, 0.0499, 0.0619, 0.1139]
std   = [0.0453, 0.0311, 0.0496, 0.0286, 0.0245, 0.0569]
times = [12.30,  1.19,   3.74,   3.28,  37.50,   3.42]

x = np.arange(len(methods))
width = 0.55

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("T=1000 / stride=10  —  6 Methods, 20 seeds", fontsize=13)

colors = ["#4C72B0", "#55A868", "#C44E52", "#E08B3A", "#8172B2", "#CCB974"]

# ── RMSE ──────────────────────────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(x, rmse, width, yerr=std, capsize=5, color=colors, alpha=0.85)
ax.set_title("Mean RMSE (± 1 std)")
ax.set_ylabel("RMSE")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.set_ylim(0, max(rmse) * 1.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
for bar, val in zip(bars, rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# ── Time ──────────────────────────────────────────────────────────────────
ax = axes[1]
bars = ax.bar(x, times, width, color=colors, alpha=0.85)
ax.set_title("Mean Inference Time per Seed (s)")
ax.set_ylabel("Seconds")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.set_ylim(0, max(times) * 1.35)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out = "C:/Users/Josep/Documents/GitHub/DiffusionSummer2026/Repaint vs DPS/outputs/T1000_4methods/bar_chart_6methods.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
