#!/usr/bin/env bash
# train_all_losses_800.sh
# Trains 7 models — one per loss function — each for 800 epochs with
# loss weight = 1.0.  Saves checkpoints under:
#   ~/ocean_diffusion/Model Parameters/loss_comparison_800/
#
# Run from the server with:
#   bash ~/ocean_diffusion/train_all_losses_800.sh
# or with tmux:
#   tmux new-session -d -s loss_800 'bash ~/ocean_diffusion/train_all_losses_800.sh > ~/ocean_diffusion/loss_comparison_800.log 2>&1'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/DDPM"
SAVE_DIR="$SCRIPT_DIR/Model Parameters/loss_comparison_800"
PICKLE="$SCRIPT_DIR/data.pickle"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# ── shared hyperparameters ───────────────────────────────────────────────────
EPOCHS=800
BATCH=32
LR=2e-4
BASE_CH=64
TIME_DIM=256
T=1000
SCHEDULE=cosine

echo "========================================================"
echo " All-loss training (800 epochs, w=1.0) — $(date)"
echo " Saving to: $SAVE_DIR"
echo "========================================================"

run_training() {
    local label="$1"
    shift
    local extra_args=("$@")

    # Skip only if checkpoint exists AND was trained for >= 800 epochs
    local tag="ddpm_${label}_gaussian_${SCHEDULE}"
    local ckpt="$SAVE_DIR/model_${tag}.pt"
    if [ -f "$ckpt" ]; then
        local trained_epochs
        trained_epochs=$(python3 - <<EOF
import torch, sys
try:
    ck = torch.load("$ckpt", map_location="cpu", weights_only=False)
    print(ck.get("epoch", 0))
except Exception as e:
    print(0)
EOF
)
        if [ "$trained_epochs" -ge "$EPOCHS" ] 2>/dev/null; then
            echo "Skipping $label — model_${tag}.pt already trained for ${trained_epochs} epochs"
            return 0
        else
            local backup="${ckpt%.pt}_${trained_epochs}ep.pt"
            echo "Re-training $label — backing up ${trained_epochs}-epoch checkpoint to $(basename "$backup")"
            mv "$ckpt" "$backup"
        fi
    fi

    echo ""
    echo "--------------------------------------------------------"
    echo " Starting: $label  $(date)"
    echo "--------------------------------------------------------"

    cd "$SCRIPT_DIR"
    PYTHONPATH="$SCRIPT_DIR" python3 "$TRAIN_DIR/train.py" \
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

# 2 ── Curl/divergence regularisation  (λ = 1.0)
run_training "curl_div" \
    --loss curl_div --weights 1.0

# 3 ── Spectral power-spectrum loss  (λ = 1.0)
run_training "spectral" \
    --loss spectral --weights 1.0

# 4 ── Okubo-Weiss parameter loss  (λ = 1.0)
run_training "okubo_weiss" \
    --loss okubo_weiss --weights 1.0

# 5 ── Sinkhorn-Wasserstein vorticity loss  (λ = 1.0)
run_training "wasserstein" \
    --loss wasserstein --weights 1.0

# 6 ── Stream-function loss  (λ = 1.0)
run_training "stream_function" \
    --loss stream_function --weights 1.0

# 7 ── Strain-rate tensor loss  (λ = 1.0)
run_training "strain_rate" \
    --loss strain_rate --weights 1.0

echo ""
echo "========================================================"
echo " All 7 runs complete — $(date)"
echo " Checkpoints in: $SAVE_DIR"
echo "========================================================"
