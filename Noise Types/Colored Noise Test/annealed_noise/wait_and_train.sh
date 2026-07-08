#!/bin/bash
# wait_and_train_annealed.sh
# Waits for pink_noise_full and red_noise_full to finish (PIDs passed as args),
# then starts annealed_noise training from scratch.
#
# Usage: bash wait_and_train_annealed.sh <PID1> <PID2>

PID1=${1:-0}
PID2=${2:-0}
LOG="/root/DiffusionSummer2026/Colored Noise Test/annealed_noise/train.log"
PICKLE="/root/model_pink_noise/data.pickle"
TRAIN_SCRIPT="/root/DiffusionSummer2026/Colored Noise Test/annealed_noise/train.py"

echo "[$(date)] Waiting for PID $PID1 and PID $PID2 to finish..."

while true; do
    p1_alive=0
    p2_alive=0
    kill -0 "$PID1" 2>/dev/null && p1_alive=1
    kill -0 "$PID2" 2>/dev/null && p2_alive=1

    if [ $p1_alive -eq 0 ] && [ $p2_alive -eq 0 ]; then
        echo "[$(date)] Both jobs finished. Starting annealed_noise training..."
        break
    fi

    echo "[$(date)] Still waiting (PID $PID1 alive=$p1_alive, PID $PID2 alive=$p2_alive)..."
    sleep 60
done

cd /root/DiffusionSummer2026
nohup python3 -u "$TRAIN_SCRIPT" \
    --pickle "$PICKLE" \
    --epochs 500 \
    --save_every 100 \
    --patience 25 \
    > "$LOG" 2>&1 &
echo "[$(date)] annealed_noise training started with PID $!"
