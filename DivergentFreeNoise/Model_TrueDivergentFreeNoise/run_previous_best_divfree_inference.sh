#!/usr/bin/env bash
set -e

cd "/root/Model_TrueDivergentFreeNoise/Basic DDPM"
python3 visualize_infer.py \
  --checkpoint "/root/Model_TrueDivergentFreeNoise/Basic DDPM/checkpoints/conditional_inpainting_2026-06-14/checkpoints/best_model.pt" \
  --pickle ../data_divfree.pickle \
  --sample 3 \
  --resample 10 \
  --output-dir testing/previous_best_divfree