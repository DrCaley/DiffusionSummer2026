#!/bin/bash
# launch_full_noise_training.sh
# Launches pink_noise_full and red_noise_full training jobs.
# Run from: /root/DiffusionSummer2026

PICKLE="/root/model_pink_noise/data.pickle"
BASE="/root/DiffusionSummer2026/Colored Noise Test"
PINK_DIR="$BASE/pink_noise_full"
RED_DIR="$BASE/red_noise_full"
PINK_CKPT="$PINK_DIR/checkpoints/best_model.pt"
RED_CKPT="$RED_DIR/checkpoints/best_model.pt"

# Determine resume flags
PINK_RESUME=""
if [ -f "$PINK_CKPT" ]; then
    PINK_RESUME="--resume $PINK_CKPT"
    echo "pink_noise_full: resuming from $PINK_CKPT"
else
    echo "pink_noise_full: starting fresh"
fi

RED_RESUME=""
if [ -f "$RED_CKPT" ]; then
    RED_RESUME="--resume $RED_CKPT"
    echo "red_noise_full: resuming from $RED_CKPT"
else
    echo "red_noise_full: starting fresh"
fi

# Launch pink_noise_full
if [ -n "$PINK_RESUME" ]; then
    nohup python3 -u "$PINK_DIR/train.py" \
        --pickle "$PICKLE" --epochs 500 --save_every 100 --patience 25 \
        --resume "$PINK_CKPT" \
        > "$PINK_DIR/train.log" 2>&1 &
else
    nohup python3 -u "$PINK_DIR/train.py" \
        --pickle "$PICKLE" --epochs 500 --save_every 100 --patience 25 \
        > "$PINK_DIR/train.log" 2>&1 &
fi
echo "pink_noise_full PID: $!"

# Launch red_noise_full
if [ -n "$RED_RESUME" ]; then
    nohup python3 -u "$RED_DIR/train.py" \
        --pickle "$PICKLE" --epochs 500 --save_every 100 --patience 25 \
        --resume "$RED_CKPT" \
        > "$RED_DIR/train.log" 2>&1 &
else
    nohup python3 -u "$RED_DIR/train.py" \
        --pickle "$PICKLE" --epochs 500 --save_every 100 --patience 25 \
        > "$RED_DIR/train.log" 2>&1 &
fi
echo "red_noise_full PID: $!"
