#!/bin/bash
# Train DDPM: eps loss, cosine schedule, noise_scale=0.12, 100 epochs.
# Same config as the 800-epoch run but stopped early to avoid overfitting.
# Best model from the 800-epoch run was epoch 138, so 100 epochs should be near-optimal.
#
# Saves to DDPM/models_100ep/ to avoid overwriting the existing checkpoint.

set -e
cd /root/ocean_diffusion

mkdir -p DDPM/models_100ep DDPM/logs

PYTHONPATH=/root/ocean_diffusion python3.12 DDPM/train.py \
    --pickle     data.pickle \
    --epochs     100         \
    --noise_scale 0.12       \
    --schedule   cosine      \
    --loss       eps         \
    --save_dir   DDPM/models_100ep \
    2>&1 | tee DDPM/logs/train_ns012_100ep.log
