#!/usr/bin/env bash
# train_loss_comparison.sh
# Trains 5 models — pure MSE and each of the 4 auxiliary loss functions at
# their default weights — and saves all checkpoints under:
#   ~/ocean_diffusion/Model Parameters/loss_comparison/
#
# Run from the server with:
#   bash ~/ocean_diffusion/train_loss_comparison.sh
# or with nohup to keep it running after disconnect:
#   nohup bash ~/ocean_diffusion/train_loss_comparison.sh > ~/ocean_diffusion/loss_comparison.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR"
SAVE_DIR="$SCRIPT_DIR/Model Parameters/loss_comparison"
PICKLE="$SCRIPT_DIR/data.pickle"
LOG_DIR="$SCRIPT_DIR/Model Parameters/loss_comparison/logs"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# ── shared hyperparameters ───────────────────────────────────────────────────
EPOCHS=200
BATCH=32
LR=2e-4
BASE_CH=64
TIME_DIM=256
T=1000
SCHEDULE=cosine

echo "========================================================"
echo " Loss comparison training — $(date)"
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

    python3 "$TRAIN_DIR/train.py" \
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

# 1 ── Pure epsilon-MSE baseline
run_training "eps" \
    --loss eps

# 2 ── MSE + curl/divergence regularisation  (λ = 0.0002)
run_training "curl_div" \
    --loss curl_div

# 3 ── MSE + spectral power-spectrum loss  (λ = 0.0002)
run_training "spectral" \
    --loss spectral

# 4 ── MSE + Okubo-Weiss parameter loss  (λ = 0.001)
run_training "okubo_weiss" \
    --loss okubo_weiss

# 5 ── MSE + Sinkhorn-Wasserstein vorticity loss  (λ = 1.0)
run_training "wasserstein" \
    --loss wasserstein

echo ""
echo "========================================================"
echo " All 5 runs complete — $(date)"
echo " Checkpoints in: $SAVE_DIR"
echo "========================================================"
