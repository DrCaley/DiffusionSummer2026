#!/usr/bin/env bash
set -euo pipefail

PID_FILE='/root/Colored_Noise_DDPM/train_pid.txt'
TRAIN_CMD='python3 train.py --pickle ../data_divfree.pickle --epochs 400 --batch 32 --save-every 100'

echo "[$(date -u +%Y-%m-%d\ %H:%M:%S)] Waiting for colored-noise PID file: ${PID_FILE}"

while true; do
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" | tr -d '[:space:]')"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      echo "[$(date -u +%Y-%m-%d\ %H:%M:%S)] Waiting for colored-noise PID ${PID} to finish..."
      while kill -0 "$PID" 2>/dev/null; do
        sleep 120
        echo "[$(date -u +%Y-%m-%d\ %H:%M:%S)] Still waiting for colored-noise PID ${PID} ..."
      done
      break
    fi
  fi
  sleep 120
done

cd "/root/Model_TrueDivergentFreeNoise/Basic DDPM"
mkdir -p checkpoints
nohup python3 train.py --pickle ../data_divfree.pickle --epochs 400 --batch 32 --save-every 100 > checkpoints/true_divfree_train.log 2>&1 &
echo $! > /root/Model_TrueDivergentFreeNoise/true_divfree_train.pid
echo "[$(date -u +%Y-%m-%d\ %H:%M:%S)] Launched True Divergent Free Noise training with PID $(cat /root/Model_TrueDivergentFreeNoise/true_divfree_train.pid)"
