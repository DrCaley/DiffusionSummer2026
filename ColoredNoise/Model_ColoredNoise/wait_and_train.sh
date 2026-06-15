#!/bin/bash
EXISTING_PID=37963
LOG=/root/Colored_Noise_DDPM/DDPM/checkpoints/train.log
mkdir -p /root/Colored_Noise_DDPM/DDPM/checkpoints

echo "[$(date)] Waiting for PID $EXISTING_PID (Divergence_Free_DDPM) to finish before starting Colored Gaussian Noise DDPM..."
while kill -0 $EXISTING_PID 2>/dev/null; do
    sleep 120
    echo "[$(date)] Still waiting for PID $EXISTING_PID ..."
done
echo "[$(date)] PID $EXISTING_PID finished. Launching Colored Gaussian Noise DDPM training..."

cd /root/Colored_Noise_DDPM/DDPM
nohup python3 train.py \
    --pickle ../../data.pickle \
    --epochs 1500 \
    --batch 32 \
    --workers 4 \
    > "$LOG" 2>&1 &

NEW_PID=$!
echo "[$(date)] Training started with PID=$NEW_PID. Log: $LOG"
echo "PID=$NEW_PID" > /root/Colored_Noise_DDPM/train_pid.txt
