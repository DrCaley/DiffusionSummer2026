#!/usr/bin/env bash
set -euo pipefail

cd "/root/Model_TrueDivergentFreeNoise/Basic DDPM"
mkdir -p checkpoints
nohup python3 train.py --pickle ../data_divfree.pickle --epochs 400 --batch 32 --save-every 100 > checkpoints/true_divfree_train.log 2>&1 &
echo $! > /root/Model_TrueDivergentFreeNoise/true_divfree_train.pid
