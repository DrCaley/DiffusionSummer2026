#!/usr/bin/env bash
set -e

mkdir -p "/root/Model_TrueDivergentFreeNoise/Basic DDPM/checkpoints/x0_hybrid_inpainting_2026-06-15"
cd "/root/Model_TrueDivergentFreeNoise/Basic DDPM"
nohup python3 train.py --epochs 400 --batch 32 --lr 2e-4 --timesteps 1000 --noise-type divergence_free --save-every 100 --run-dir checkpoints/x0_hybrid_inpainting_2026-06-15 --path-steps 150 --prediction-type x0 --reconstruction-loss-weight 1.0 --noise-loss-weight 0.25 > "checkpoints/x0_hybrid_inpainting_2026-06-15/train_stdout.log" 2>&1 &
echo $! > "/root/Model_TrueDivergentFreeNoise/Basic DDPM/checkpoints/x0_hybrid_inpainting_2026-06-15/train.pid"
echo TRAIN_PID:$!