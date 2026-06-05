#!/bin/bash
# Train Repaint UNet under all four noise schedules.
# Run from workspace root: bash NoiseSchedule/run_repaint.sh
# Mirrors run_voronoi.sh — logs go to NoiseSchedule/{schedule}_out.txt
cd /root

for SCHED in linear cosine quadratic sigmoid; do
    echo "========================================"
    echo "=== Training schedule: $SCHED ==="
    echo "========================================"
    python3 "NoiseSchedule/train_repaint.py" \
        --pickle data.pickle \
        --epochs 100 \
        --batch 32 \
        --schedule $SCHED \
        2>&1 | tee "NoiseSchedule/${SCHED}_out.txt"
done

echo ""
echo "All schedules trained."
