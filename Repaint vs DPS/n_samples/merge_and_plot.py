"""
Merge two 50-seed ensemble_nsample result files into a combined 100-seed report,
then regenerate the bar graphs.

Usage (from the n_samples folder):
    python merge_and_plot.py
    python merge_and_plot.py --r1 ensemble_nsample_results.txt --r2 ensemble_nsample_results_shift20.txt
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parse a results .txt file
# ---------------------------------------------------------------------------

def parse_results(path):
    """
    Returns:
        meta  : dict of header key/value strings
        summary: dict  n -> {"mean_time", "std_time", "mean_rmse", "std_rmse", "min_rmse", "max_rmse"}
        seeds : list of dicts  {"test_idx": int, "times": {n: float}, "rmses": {n: float}}
    """
    summary_pat = re.compile(
        r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )
    # header of per-seed section: "  test_idx  t(n=1)  RMSE(n=1)  ..."
    seed_header_pat = re.compile(r"test_idx")
    # per-seed row:  "   0     2.1s  0.0797     7.4s  0.0612  ..."
    seed_row_pat = re.compile(r"^\s*(\d+)((?:\s+[\d.]+s\s+[\d.]+)+)\s*$")

    meta = {}
    summary = {}
    seeds = []
    n_order = []
    in_seed_section = False

    with open(path) as f:
        for line in f:
            # Header meta lines
            m = re.match(r"^(Checkpoint|T=|noise_std|schedule|n_seeds|steps/run)\s*[=:]\s*(.+)", line.strip())
            if m:
                meta[m.group(1)] = m.group(2).strip()
                continue

            # Summary table rows
            m = summary_pat.match(line)
            if m and not in_seed_section:
                n = int(m.group(1))
                n_order.append(n)
                summary[n] = {
                    "mean_time": float(m.group(2)),
                    "std_time":  float(m.group(3)),
                    "mean_rmse": float(m.group(4)),
                    "std_rmse":  float(m.group(5)),
                    "min_rmse":  float(m.group(6)),
                    "max_rmse":  float(m.group(7)),
                }
                continue

            if seed_header_pat.search(line):
                in_seed_section = True
                continue

            if in_seed_section:
                m = seed_row_pat.match(line)
                if m:
                    test_idx = int(m.group(1))
                    tokens = m.group(2).split()
                    # tokens: "2.1s", "0.0797", "7.4s", "0.0612", ...
                    times = {}
                    rmses = {}
                    for i, n in enumerate(n_order):
                        t_str = tokens[2 * i].rstrip("s")
                        r_str = tokens[2 * i + 1]
                        times[n] = float(t_str)
                        rmses[n] = float(r_str)
                    seeds.append({"test_idx": test_idx, "times": times, "rmses": rmses})

    return meta, summary, seeds, n_order


# ---------------------------------------------------------------------------
# Combine two result sets
# ---------------------------------------------------------------------------

def combine(seeds1, seeds2, n_order):
    """Pool all per-seed records and recompute summary statistics."""
    all_seeds = seeds1 + seeds2
    combined_summary = {}
    for n in n_order:
        times = [s["times"][n] for s in all_seeds]
        rmses = [s["rmses"][n] for s in all_seeds]
        combined_summary[n] = {
            "mean_time": float(np.mean(times)),
            "std_time":  float(np.std(times)),
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse":  float(np.std(rmses)),
            "min_rmse":  float(np.min(rmses)),
            "max_rmse":  float(np.max(rmses)),
        }
    return combined_summary, all_seeds


# ---------------------------------------------------------------------------
# Write combined report
# ---------------------------------------------------------------------------

def write_report(combined_summary, all_seeds, n_order, meta, out_path):
    lines = []
    lines.append("Ensemble n_samples experiment  [COMBINED 100-seed report]")
    for k, v in meta.items():
        lines.append(f"{k:<14}: {v}")
    lines.append(f"n_seeds       : {len(all_seeds)}")
    lines.append("")
    lines.append(f"{'n_samples':>10}  {'mean_time(s)':>13}  {'std_time':>9}  "
                 f"{'mean_RMSE':>10}  {'std_RMSE':>9}  {'min_RMSE':>9}  {'max_RMSE':>9}")
    lines.append("-" * 80)
    for n in n_order:
        s = combined_summary[n]
        lines.append(
            f"{n:>10}  {s['mean_time']:>13.2f}  {s['std_time']:>9.2f}  "
            f"{s['mean_rmse']:>10.4f}  {s['std_rmse']:>9.4f}  "
            f"{s['min_rmse']:>9.4f}  {s['max_rmse']:>9.4f}"
        )
    lines.append("")
    lines.append("Per-seed breakdown:")
    header = f"{'test_idx':>10}" + "".join(f"  t(n={n})  RMSE(n={n})" for n in n_order)
    lines.append(header)
    lines.append("-" * (10 + 22 * len(n_order)))
    for seed in sorted(all_seeds, key=lambda s: s["test_idx"]):
        row = f"{seed['test_idx']:>10}"
        for n in n_order:
            row += f"  {seed['times'][n]:6.1f}s  {seed['rmses'][n]:.4f}"
        lines.append(row)

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"Saved report : {out_path}")


# ---------------------------------------------------------------------------
# Bar plot
# ---------------------------------------------------------------------------

def make_bar_plot(summary50, summary100, n_order, out_path):
    n_arr = np.array(n_order)
    x = np.arange(len(n_arr))
    width = 0.35

    mr50  = np.array([summary50[n]["mean_rmse"]  for n in n_order])
    sr50  = np.array([summary50[n]["std_rmse"]   for n in n_order])
    mr100 = np.array([summary100[n]["mean_rmse"] for n in n_order])
    sr100 = np.array([summary100[n]["std_rmse"]  for n in n_order])

    mt50  = np.array([summary50[n]["mean_time"]  for n in n_order])
    st50  = np.array([summary50[n]["std_time"]   for n in n_order])
    mt100 = np.array([summary100[n]["mean_time"] for n in n_order])
    st100 = np.array([summary100[n]["std_time"]  for n in n_order])

    ecolor = "#1a1a2e"
    ekw = {"ecolor": ecolor, "lw": 1.5}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Ensemble n_samples: 50-seed vs 100-seed comparison\n"
        "T=1000, stride=10, resample=1",
        fontsize=13, fontweight="bold",
    )

    # ---- RMSE ----
    ax = axes[0]
    b50  = ax.bar(x - width/2, mr50,  width, yerr=sr50,  capsize=4, color="#4C8FBF",
                  edgecolor="white", label="50 seeds", error_kw=ekw)
    b100 = ax.bar(x + width/2, mr100, width, yerr=sr100, capsize=4, color="#2a5f8f",
                  edgecolor="white", label="100 seeds", error_kw=ekw)
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in n_order], fontsize=11)
    ax.set_xlabel("n_samples (ensemble size)", fontsize=11)
    ax.set_ylabel("Mean RMSE (±1 std)", fontsize=11)
    ax.set_title("RMSE vs Ensemble Size", fontsize=12)
    ax.set_ylim(0, max(np.max(mr50 + sr50), np.max(mr100 + sr100)) * 1.25)
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(b100, mr100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # ---- Time ----
    ax = axes[1]
    ax.bar(x - width/2, mt50,  width, yerr=st50,  capsize=4, color="#E07B54",
           edgecolor="white", label="50 seeds", error_kw=ekw)
    bt100 = ax.bar(x + width/2, mt100, width, yerr=st100, capsize=4, color="#a04820",
                   edgecolor="white", label="100 seeds", error_kw=ekw)
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in n_order], fontsize=11)
    ax.set_xlabel("n_samples (ensemble size)", fontsize=11)
    ax.set_ylabel("Mean inference time (s, ±1 std)", fontsize=11)
    ax.set_title("Inference Time vs Ensemble Size", fontsize=12)
    ax.set_ylim(0, max(np.max(mt50 + st50), np.max(mt100 + st100)) * 1.25)
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bt100, mt100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure : {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--r1", default="ensemble_nsample_results.txt")
    p.add_argument("--r2", default="ensemble_nsample_results_shift20.txt")
    p.add_argument("--out_txt", default="ensemble_nsample_results_100seeds.txt")
    p.add_argument("--out_png", default="nsample_bar_graph_100seeds.png")
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))

    r1_path = os.path.join(base, args.r1)
    r2_path = os.path.join(base, args.r2)

    print(f"Loading {r1_path}")
    meta1, summary50_a, seeds1, n_order = parse_results(r1_path)
    print(f"  -> {len(seeds1)} seeds, n_order={n_order}")

    print(f"Loading {r2_path}")
    _, summary50_b, seeds2, _ = parse_results(r2_path)
    print(f"  -> {len(seeds2)} seeds")

    # Average of the two 50-seed summaries (for comparison bars)
    summary50_avg = {}
    for n in n_order:
        all_rmses = [s["rmses"][n] for s in seeds1 + seeds2]
        all_times = [s["times"][n] for s in seeds1 + seeds2]
        summary50_avg[n] = {
            "mean_rmse": summary50_a[n]["mean_rmse"],
            "std_rmse":  summary50_a[n]["std_rmse"],
            "mean_time": summary50_a[n]["mean_time"],
            "std_time":  summary50_a[n]["std_time"],
        }

    combined_summary, all_seeds = combine(seeds1, seeds2, n_order)
    print(f"Combined: {len(all_seeds)} seeds total")

    write_report(combined_summary, all_seeds, n_order, meta1,
                 os.path.join(base, args.out_txt))

    make_bar_plot(summary50_a, combined_summary, n_order,
                  os.path.join(base, args.out_png))


if __name__ == "__main__":
    main()
