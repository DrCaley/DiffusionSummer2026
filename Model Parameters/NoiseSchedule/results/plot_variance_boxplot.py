"""
Box plot of RMSE variance across noise schedules.
Reads existing batch_{schedule}.log files — no new inference needed.

Log files are expected at:
    NoiseSchedule/{schedule}/model_{schedule}_results/batch_{schedule}.log

Usage (run from workspace root):
    py NoiseSchedule/plot_variance_boxplot.py
    py NoiseSchedule/plot_variance_boxplot.py --schedules cosine cosine_s0001 cosine_s02 cosine_s10
    py NoiseSchedule/plot_variance_boxplot.py --out my_plot.png

All available schedules:
    linear, cosine, cosine_s0001, cosine_s02, cosine_s10,
    quadratic, sigmoid, geometric
"""

import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np

ALL_SCHEDULES = [
    "linear", "cosine", "cosine_s0001", "cosine_s02", "cosine_s10",
    "quadratic", "sigmoid", "geometric",
]

COLORS = {
    "linear":       "#d62728",
    "cosine":       "#1f77b4",
    "cosine_s0001": "#aec7e8",
    "cosine_s02":   "#6baed6",
    "cosine_s10":   "#08519c",
    "quadratic":    "#2ca02c",
    "sigmoid":      "#ff7f0e",
    "geometric":    "#9467bd",
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--schedules", nargs="+", default=ALL_SCHEDULES,
               choices=ALL_SCHEDULES, metavar="SCHEDULE",
               help="Schedules to include (default: all). "
                    f"Choices: {', '.join(ALL_SCHEDULES)}")
p.add_argument("--out", default=None,
               help="Output PNG path. Defaults to variance_results/boxplot_variance.png")
args = p.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Parse logs
# ---------------------------------------------------------------------------
data = {}
missing = []
for sched in args.schedules:
    log = os.path.join(script_dir, sched, f"model_{sched}_results", f"batch_{sched}.log")
    if not os.path.exists(log):
        print(f"WARNING: log not found for '{sched}': {log}")
        missing.append(sched)
        continue
    rmses = []
    with open(log) as f:
        for line in f:
            m = re.search(r"RMSE\s*=\s*([\d.]+)", line)
            if m:
                rmses.append(float(m.group(1)))
    data[sched] = rmses
    print(f"{sched:15s}: n={len(rmses)}  mean={np.mean(rmses):.4f}  std={np.std(rmses):.4f}")

schedules = [s for s in args.schedules if s in data]
if not schedules:
    print("No data found — nothing to plot.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Box plot + strip plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(max(8, len(schedules) * 1.4), 6))

positions = list(range(len(schedules)))
box_data  = [data[s] for s in schedules]

bp = ax.boxplot(box_data, positions=positions, widths=0.4,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=2))

for patch, sched in zip(bp["boxes"], schedules):
    patch.set_facecolor(COLORS[sched])
    patch.set_alpha(0.6)

rng = np.random.default_rng(42)
for i, sched in enumerate(schedules):
    jitter = rng.uniform(-0.15, 0.15, len(data[sched]))
    ax.scatter(np.full(len(data[sched]), i) + jitter, data[sched],
               color=COLORS[sched], edgecolors="black", linewidths=0.5,
               s=40, zorder=3, alpha=0.85)

for i, sched in enumerate(schedules):
    vals = data[sched]
    ax.text(i, max(vals) + 0.01,
            f"μ={np.mean(vals):.4f}\nσ={np.std(vals):.4f}",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(positions)
ax.set_xticklabels(schedules, fontsize=11, rotation=15, ha="right")
ax.set_ylabel("RMSE", fontsize=12)
ax.set_title("RePaint RMSE distribution across 10 val samples\n(each point = one sample+path run)",
             fontsize=13)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()

out_dir = os.path.join(script_dir, "variance_results")
os.makedirs(out_dir, exist_ok=True)
out = args.out or os.path.join(out_dir, "boxplot_variance.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved: {out}")
