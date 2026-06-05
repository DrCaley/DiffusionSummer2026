#!/bin/bash
# Train Repaint UNet under all four noise schedules.
# Run from workspace root: bash "Model Parameters/NoiseSchedule/run_repaint.sh"
# Logs go to Model Parameters/NoiseSchedule/checkpoints/checkpoints_repaint_{schedule}/{schedule}_out.txt
cd /root/ocean_diffusion

for SCHED in linear cosine quadratic sigmoid; do
    echo "========================================"
    echo "=== Training schedule: $SCHED ==="
    echo "========================================"
    LOG_DIR="Model Parameters/NoiseSchedule/checkpoints/checkpoints_repaint_${SCHED}"
    mkdir -p "$LOG_DIR"
    python3 "Model Parameters/NoiseSchedule/train_repaint.py" \
        --pickle data.pickle \
        --epochs 100 \
        --batch 32 \
        --schedule $SCHED \
        2>&1 | tee "${LOG_DIR}/${SCHED}_out.txt"
done

echo ""
echo "All schedules trained."
