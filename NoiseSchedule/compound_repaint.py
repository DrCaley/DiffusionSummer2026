"""
Compound repaint comparison figures.

For each of the 10 val samples, reads the existing batch result PNGs from
all 5 schedules and builds two 7-panel compound images:

  compound_recon_{run:02d}.png  — GT | Robot Path | linear | cosine | quadratic | sigmoid | geometric  (reconstructions)
  compound_error_{run:02d}.png  — GT | Robot Path | linear | cosine | quadratic | sigmoid | geometric  (error maps)

Layout: 2 rows x 4 cols (last cell empty)
  Row 0: GT | RobotPath | linear | cosine
  Row 1: quadratic | sigmoid | geometric | [empty]

Usage (run from workspace root):
    py NoiseSchedule/compound_repaint.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

SCHEDULES   = ["linear", "cosine", "quadratic", "sigmoid", "geometric"]
SCHED_LABELS= ["Linear", "Cosine", "Quadratic", "Sigmoid", "Geometric"]
N_RUNS      = 10

# Pixel crop coordinates detected from 2700x1500 source images
# (left, upper, right, lower) — PIL convention (y=0 at top)
SPLIT_Y  = 706   # horizontal split between subplot rows
SPLIT_X  = 1104  # vertical split between subplot columns
TOP_Y    = 30    # top of content (below suptitle)
BOT_Y    = 1500  # bottom of image

GT_CROP    = (0,       TOP_Y,   SPLIT_X, SPLIT_Y)  # top-left
PATH_CROP  = (SPLIT_X, TOP_Y,   2700,    SPLIT_Y)  # top-right
RECON_CROP = (0,       SPLIT_Y, SPLIT_X, BOT_Y)    # bottom-left
ERROR_CROP = (SPLIT_X, SPLIT_Y, 2700,    BOT_Y)    # bottom-right

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "models_compounded_results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_crop(folder, run, box):
    path = os.path.join(SCRIPT_DIR, folder, f"result_{run:02d}.png")
    return Image.open(path).crop(box)


def make_compound(run, img_type, panels, labels, sample_idx, seed):
    """
    panels : list of 7 PIL Images [GT, RobotPath, sched0..4]
    img_type: "recon" or "error"
    """
    type_title = "Reconstruction" if img_type == "recon" else "Error Map"
    positions  = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]

    fig = plt.figure(figsize=(36, 12))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.12, wspace=0.04)

    for (r, c), img, label in zip(positions, panels, labels):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(np.array(img))
        ax.set_title(label, fontsize=12,
                     fontweight="bold" if label in ("Ground Truth","Robot Path") else "normal")
        ax.axis("off")

    # empty last cell
    fig.add_subplot(gs[1, 3]).axis("off")

    fig.suptitle(
        f"Compound {type_title}  —  Val sample {sample_idx}, seed {seed}  "
        f"[all 5 schedules]",
        fontsize=14,
    )

    out = os.path.join(OUT_DIR, f"compound_{img_type}_{run:02d}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


for run in range(1, N_RUNS + 1):
    sample_idx = run - 1
    seed       = sample_idx * 7 + 1
    print(f"\n[{run:02d}/10]  sample={sample_idx}, seed={seed}")

    # GT and Robot Path are identical across schedules — take from cosine
    ref_folder = "model_cosine_results"
    gt_img   = load_crop(ref_folder, run, GT_CROP)
    path_img = load_crop(ref_folder, run, PATH_CROP)

    recon_imgs = [load_crop(f"model_{s}_results", run, RECON_CROP) for s in SCHEDULES]
    error_imgs = [load_crop(f"model_{s}_results", run, ERROR_CROP) for s in SCHEDULES]

    panel_labels = ["Ground Truth", "Robot Path"] + SCHED_LABELS

    make_compound(run, "recon", [gt_img, path_img] + recon_imgs, panel_labels, sample_idx, seed)
    make_compound(run, "error", [gt_img, path_img] + error_imgs, panel_labels, sample_idx, seed)

print("\nAll done.")
