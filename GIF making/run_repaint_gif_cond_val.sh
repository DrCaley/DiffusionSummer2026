#!/bin/bash
# Run repaint_gif_cond.py on 10 random validation samples using the
# voronoi-conditioned DDPM (noise_scale=0.12).
# Usage:
#   bash run_repaint_gif_cond_val.sh [--resample 10] [--capture_every 20]
# All extra args are forwarded to repaint_gif_cond.py.

set -e
cd /root/ocean_diffusion

LOGDIR="${REPAINT_LOGDIR:-repaint_gif_cond_val_logs}"
mkdir -p "$LOGDIR"

# Clear stale pycache to avoid import conflicts
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Pick 10 random indices from the validation set
INDICES=$(python3.12 - <<'EOF'
import pickle, random
with open("data.pickle", "rb") as f:
    data = pickle.load(f)
n = data[1].shape[3]   # split=1, last dim = N samples
random.seed(42)
print(" ".join(str(i) for i in random.sample(range(n), min(10, n))))
EOF
)

echo "Validation set sample indices: $INDICES"
echo "Logging to: $LOGDIR/"

for IDX in $INDICES; do
    OUT="$LOGDIR/repaint_val_sample${IDX}.gif"
    LOG="$LOGDIR/repaint_val_sample${IDX}.log"
    echo "Running sample $IDX -> $OUT"
    python3.12 repaint_gif_cond.py \
        --split  1 \
        --sample "$IDX" \
        --seed   "$IDX" \
        --out    "$OUT" \
        "$@" \
        > "$LOG" 2>&1
    echo "  done (log: $LOG)"
done

echo ""
echo "All 10 conditional repaint GIFs saved to $LOGDIR/"

# ---- RMSE summary -------------------------------------------------------
echo ""
echo "=== RMSE Summary ==="
TOTAL=0; COUNT=0
for IDX in $INDICES; do
    LOG="$LOGDIR/repaint_val_sample${IDX}.log"
    RMSE=$(grep "^RMSE" "$LOG" 2>/dev/null | awk '{print $3}')
    if [ -n "$RMSE" ]; then
        echo "  sample $IDX : $RMSE"
        TOTAL=$(python3.12 -c "print($TOTAL + $RMSE)")
        COUNT=$((COUNT + 1))
    else
        echo "  sample $IDX : ERROR (see $LOG)"
    fi
done
if [ "$COUNT" -gt 0 ]; then
    MEAN=$(python3.12 -c "print(f'{$TOTAL / $COUNT:.6f}')")
    echo "  ----------------------"
    echo "  mean RMSE  : $MEAN  ($COUNT samples)"
fi
