#!/bin/bash
# Train conditional DDPM (voronoi mode) with noise_scale=0.12
set -e
cd /root/ocean_diffusion

LOG="Conditional DDPM/train_voronoi_ns012.log"
echo "Starting voronoi ns=0.12 training — logging to $LOG"

python3.12 "Conditional DDPM/train.py" \
    --cond        voronoi \
    --noise_scale 0.12    \
    --epochs      400     \
    --batch       16      \
    --pickle      data.pickle \
    --save_dir    "Conditional DDPM/checkpoints_voronoi_ns012" \
    2>&1 | tee "$LOG"

echo "Training complete."
