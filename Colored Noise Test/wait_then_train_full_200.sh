#!/bin/bash
# wait_then_train_full_200.sh
# Waits for the annealed_noise job to finish, then runs pink_noise_full and
# red_noise_full to epoch 200 with patience disabled (--patience 9999).
# Usage: bash wait_then_train_full_200.sh <ANNEALED_PID>

ANNEALED_PID=${1:-0}
PICKLE="/root/model_pink_noise/data.pickle"
PINK_SCRIPT="/root/DiffusionSummer2026/Colored Noise Test/pink_noise_full/train.py"
RED_SCRIPT="/root/DiffusionSummer2026/Colored Noise Test/red_noise_full/train.py"
PINK_CKPT="/root/DiffusionSummer2026/Colored Noise Test/pink_noise_full/checkpoints/best_model.pt"
RED_CKPT="/root/DiffusionSummer2026/Colored Noise Test/red_noise_full/checkpoints/best_model.pt"
PINK_LOG="/root/DiffusionSummer2026/Colored Noise Test/pink_noise_full/train.log"
RED_LOG="/root/DiffusionSummer2026/Colored Noise Test/red_noise_full/train.log"

echo "[$(date)] Waiting for annealed_noise PID $ANNEALED_PID to finish..."

while kill -0 "$ANNEALED_PID" 2>/dev/null; do
    echo "[$(date)] Still waiting for PID $ANNEALED_PID..."
    sleep 60
done

echo "[$(date)] Annealed job finished. Starting pink_noise_full and red_noise_full to epoch 200..."

# pink_noise_full — resume from best, run to epoch 200, patience=9999
nohup python3 -u "$PINK_SCRIPT" \
    --pickle "$PICKLE" \
    --epochs 200 \
    --save_every 100 \
    --patience 9999 \
    --resume "$PINK_CKPT" \
    >> "$PINK_LOG" 2>&1 &
echo "[$(date)] pink_noise_full PID: $!"

# red_noise_full — resume from best, run to epoch 200, patience=9999
nohup python3 -u "$RED_SCRIPT" \
    --pickle "$PICKLE" \
    --epochs 200 \
    --save_every 100 \
    --patience 9999 \
    --resume "$RED_CKPT" \
    >> "$RED_LOG" 2>&1 &
echo "[$(date)] red_noise_full PID: $!"
