import numpy as np
import matplotlib.pyplot as plt

methods = ["RePaint-r10", "RePaint-r1", "DPS", "RePaint+DPS"]

# T=100, stride=1
rmse_100  = [0.0878, 0.0915, 0.0910, 0.1050]
std_100   = [0.0264, 0.0387, 0.0363, 0.0212]
time_100  = [16.25,  1.56,   5.55,  57.54]

# T=1000, stride=10
rmse_1000 = [0.1055, 0.0772, 0.1033, 0.0619]
std_1000  = [0.0453, 0.0311, 0.0496, 0.0245]
time_1000 = [12.30,  1.19,   3.74,  37.50]

x = np.arange(len(methods))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Method Comparison: T=100 vs T=1000  (20 seeds, same ground truths)", fontsize=13)

# ── RMSE bar chart ──────────────────────────────────────────────────────────
ax = axes[0]
b1 = ax.bar(x - width/2, rmse_100,  width, yerr=std_100,  capsize=4,
            label="T=100 / stride=1",  color="#4C72B0", alpha=0.85)
b2 = ax.bar(x + width/2, rmse_1000, width, yerr=std_1000, capsize=4,
            label="T=1000 / stride=10", color="#DD8452", alpha=0.85)
ax.set_title("Mean RMSE (± 1 std, 20 seeds)")
ax.set_ylabel("RMSE")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, max(max(rmse_100), max(rmse_1000)) * 1.55)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# annotate bar values
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

# ── Time bar chart ──────────────────────────────────────────────────────────
ax = axes[1]
b3 = ax.bar(x - width/2, time_100,  width, label="T=100 / stride=1",
            color="#4C72B0", alpha=0.85)
b4 = ax.bar(x + width/2, time_1000, width, label="T=1000 / stride=10",
            color="#DD8452", alpha=0.85)
ax.set_title("Mean Inference Time per Seed (s)")
ax.set_ylabel("Seconds")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, max(max(time_100), max(time_1000)) * 1.35)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

for bar in b3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8)
for bar in b4:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out = "C:/Users/Josep/Documents/GitHub/DiffusionSummer2026/Repaint vs DPS/outputs/comparison_T100_vs_T1000.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
