#!/bin/bash
set -e
cd /root/DiffusionSummer2026

echo "=== Re-rendering r1_combined ==="
/root/ocean_ddpm/venv/bin/python "Colored Noise Test/batch_white_red_annealed.py" \
    --pickle /root/model_pink_noise/data.pickle \
    --ckpt best \
    --out_dir "Colored Noise Test/outputs/r1_combined"

echo "=== Re-rendering batch_best ==="
/root/ocean_ddpm/venv/bin/python "Colored Noise Test/batch_infer_all.py" \
    --pickle /root/model_pink_noise/data.pickle \
    --ckpt best \
    --out_dir "Colored Noise Test/outputs/batch_best"

echo "=== DONE ==="
