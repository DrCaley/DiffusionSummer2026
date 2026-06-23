#!/bin/bash
cd /root/ocean_diffusion
python3.12 "Conditional DDPM/compare_infer.py" \
    --pickle data.pickle \
    --eps_ckpt  checkpoints/best_model.pt \
    --cond_ckpt "Conditional DDPM/checkpoints_voronoi/best_cond_ddpm_voronoi_cosine.pt" \
    --n_samples 10 --path_steps 150 --resample 10 --seed 0 \
    > "Conditional DDPM/compare_infer.log" 2>&1
echo DONE
