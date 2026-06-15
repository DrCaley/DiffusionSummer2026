#!/usr/bin/env bash
set -e
cd "/root/Model_TrueDivergentFreeNoise/Basic DDPM"
python3 visualize_infer.py --checkpoint "checkpoints/conditional_inpainting_2026-06-14/checkpoints/final_model.pt" --pickle "../data_divfree.pickle" --sample 3 --path_steps 150 --resample 10 --output-dir "/root/Model_TrueDivergentFreeNoise"
