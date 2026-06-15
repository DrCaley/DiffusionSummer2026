#!/usr/bin/env bash
set -e
backup_dir=/root/Model_TrueDivergentFreeNoise/backups/current_model_2026-06-14
mkdir -p "$backup_dir"
if [ -d "/root/Model_TrueDivergentFreeNoise/Basic DDPM/checkpoints" ]; then
  cp -r "/root/Model_TrueDivergentFreeNoise/Basic DDPM/checkpoints" "$backup_dir/"
fi
mkdir -p "/root/Model_TrueDivergentFreeNoise/Basic DDPM/checkpoints/conditional_inpainting_2026-06-14"
cd "/root/Model_TrueDivergentFreeNoise/Basic DDPM"
nohup python3 train.py --epochs 400 --batch 32 --lr 2e-4 --timesteps 1000 --noise-type divergence_free --save-every 100 --run-dir checkpoints/conditional_inpainting_2026-06-14 --path-steps 150 > "checkpoints/conditional_inpainting_2026-06-14/train_stdout.log" 2>&1 &
echo $! > "/root/Model_TrueDivergentFreeNoise/Basic DDPM/checkpoints/conditional_inpainting_2026-06-14/train.pid"
echo TRAIN_PID:$!
