#!/usr/bin/env bash
# train_spectral_okubo.sh
# Retrains spectral and okubo_weiss models at 400 epochs, weight=1.0.
# Saves to: ~/ocean_diffusion/Model Parameters/loss_comparison_w1/
#
# tmux usage:
#   tmux new-session -d -s spec_oku 'bash ~/ocean_diffusion/train_spectral_okubo.sh > ~/ocean_diffusion/spec_oku.log 2>&1'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"   # one level up from DDPM/
TRAIN_DIR="$SCRIPT_DIR"
SAVE_DIR="$REPO_DIR/Model Parameters/loss_comparison_w1"
PICKLE="$REPO_DIR/data.pickle"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

EPOCHS=400
BATCH=32
LR=2e-4
BASE_CH=64
TIME_DIM=256
T=1000
SCHEDULE=cosine

echo "========================================================"
echo " spectral + okubo_weiss retraining (w=1.0) — $(date)"
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

    PYTHONPATH="$REPO_DIR" python3 "$TRAIN_DIR/train.py" \
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

run_training "spectral"    --loss spectral    --weights 1.0
run_training "okubo_weiss" --loss okubo_weiss --weights 1.0

echo ""
echo "========================================================"
echo " Done — $(date)"
echo "========================================================"
