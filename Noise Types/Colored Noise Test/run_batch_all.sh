#!/bin/bash
# run_batch_all.sh — run both batch_best and batch_epoch100 sequentially
BASE="/root/DiffusionSummer2026/Colored Noise Test"
PICKLE="/root/model_pink_noise/data.pickle"
PY="$BASE/batch_infer_all.py"

source /root/ocean_ddpm/venv/bin/activate
cd /root/DiffusionSummer2026

echo "==============================="
echo " BATCH 1: best checkpoints"
echo "==============================="
python3 -u "$PY" \
    --pickle "$PICKLE" \
    --ckpt best \
    --out_dir "$BASE/outputs/batch_best" \
    2>&1 | tee "$BASE/outputs/batch_best_run.log"

echo ""
echo "==============================="
echo " BATCH 2: best-by-epoch-100 checkpoints"
echo "==============================="
python3 -u "$PY" \
    --pickle "$PICKLE" \
    --ckpt best_by_100 \
    --out_dir "$BASE/outputs/batch_epoch100" \
    2>&1 | tee "$BASE/outputs/batch_epoch100_run.log"

echo ""
echo "All batches complete."
