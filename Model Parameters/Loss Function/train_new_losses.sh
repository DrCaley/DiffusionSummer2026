#!/usr/bin/env bash
# train_new_losses.sh
# Trains two new models:
#   1. stream_function loss  (weight = 1.0)
#   2. strain_rate loss      (weight = 1.0)
# Both use cosine schedule, 400 epochs, and save under:
#   ~/ocean_diffusion/Model Parameters/loss_comparison/
#
# Run from the server with:
#   nohup bash ~/ocean_diffusion/"Model Parameters/Loss Function/train_new_losses.sh" \
#     > ~/ocean_diffusion/"Model Parameters/loss_comparison/logs/new_losses.log" 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR is ~/ocean_diffusion/Model Parameters/Loss Function
OCEAN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"   # ~/ocean_diffusion
SAVE_DIR="$OCEAN_DIR/Model Parameters/loss_comparison"
PICKLE="$OCEAN_DIR/data.pickle"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# ── shared hyperparameters ───────────────────────────────────────────────────
EPOCHS=400
BATCH=32
LR=2e-4
BASE_CH=64
TIME_DIM=256
T=1000
SCHEDULE=cosine

echo "========================================================"
echo " New-loss training — $(date)"
echo " Saving to: $SAVE_DIR"
echo "========================================================"

run_training() {
    local label="$1"
    shift
    local extra_args=("$@")

    echo ""
    echo "--------------------------------------------------------"
    echo " Starting: $label  $(date)"
    echo "--------------------------------------------------------"

    python3 "$OCEAN_DIR/train.py" \
        --pickle   "$PICKLE"   \
        --epochs   "$EPOCHS"   \
        --batch    "$BATCH"    \
        --lr       "$LR"       \
        --base_ch  "$BASE_CH"  \
        --time_dim "$TIME_DIM" \
        --T        "$T"        \
        --schedule "$SCHEDULE" \
        --save_dir "$SAVE_DIR" \
        "${extra_args[@]}" \
        2>&1 | tee "$LOG_DIR/${label}.log"

    echo " Finished: $label  $(date)"
}

# 1 ── Stream-function loss  (λ = 1.0)
run_training "stream_function" \
    --loss stream_function \
    --weights 1.0

# 2 ── Strain-rate tensor loss  (λ = 1.0)
run_training "strain_rate" \
    --loss strain_rate \
    --weights 1.0

echo ""
echo "========================================================"
echo " Both runs complete — $(date)"
echo " Checkpoints in: $SAVE_DIR"
echo "========================================================"
