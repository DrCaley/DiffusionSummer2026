import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# T=1000 / stride=10, best result used for repeated methods (* = best of multiple runs)
data = [
    ("RePaint r=10*",          0.0752, 0.0302,  8.71),   # best of 3 runs
    ("RePaint r=1*",           0.0772, 0.0311,  1.19),   # best of 3 runs
    ("DPS ζ=0.5",              0.1033, 0.0496,  3.74),
    ("DPS ζ=0.04",             0.0499, 0.0286,  3.28),
    ("RePaint+DPS\nr=10 ζ=0.5",  0.0619, 0.0245, 37.50),
    ("RePaint+DPS\nr=10 ζ=0.04", 0.0618, 0.0282, 33.60),
    ("RePaint+DPS\nr=1 ζ=0.5",   0.1139, 0.0569,  3.42),
    ("RePaint+DPS\nr=1 ζ=0.04",  0.0508, 0.0210,  3.28),
]

labels = [d[0] for d in data]
rmse   = [d[1] for d in data]
std    = [d[2] for d in data]
times  = [d[3] for d in data]

# Color by method family
family_colors = {
    "RePaint r=10*":             "#4C72B0",
    "RePaint r=1*":              "#55A868",
    "DPS ζ=0.5":                 "#C44E52",
    "DPS ζ=0.04":                "#E08B3A",
    "RePaint+DPS\nr=10 ζ=0.5":   "#8172B2",
    "RePaint+DPS\nr=10 ζ=0.04":  "#937860",
    "RePaint+DPS\nr=1 ζ=0.5":    "#DA8BC3",
    "RePaint+DPS\nr=1 ζ=0.04":   "#8C8C8C",
}
colors = [family_colors[l] for l in labels]

x = np.arange(len(labels))
width = 0.55

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("T=1000 / stride=10  —  All Methods, 20 seeds  (* = best of repeated runs)",
             fontsize=10)

# ── RMSE ──────────────────────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(x, rmse, width, yerr=std, capsize=5, color=colors, alpha=0.85)
ax.set_title("Mean RMSE (± 1 std)", fontsize=10)
ax.set_ylabel("RMSE")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0, max(rmse) * 1.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
for bar, val in zip(bars, rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=7)

# ── Time ──────────────────────────────────────────────────────────────
ax = axes[1]
bars = ax.bar(x, times, width, color=colors, alpha=0.85)
ax.set_title("Mean Inference Time per Seed (s)", fontsize=10)
ax.set_ylabel("Seconds")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0, max(times) * 1.3)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
out = "C:/Users/Josep/Documents/GitHub/DiffusionSummer2026/Repaint vs DPS/outputs/T1000_bar_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
