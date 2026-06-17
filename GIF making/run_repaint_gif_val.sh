#!/bin/bash
# Run repaint_gif.py on 10 random validation samples (split=1).
# Usage:
#   bash run_repaint_gif_val.sh [--schedule cosine] [--resample 10]
# All extra args are forwarded to repaint_gif.py.

set -e
cd /root/ocean_diffusion

LOGDIR="${REPAINT_LOGDIR:-repaint_gif_val_logs}"
mkdir -p "$LOGDIR"

# Pick 10 random indices from the validation set
INDICES=$(python3.12 - <<'EOF'
import pickle, random, sys
with open("data.pickle", "rb") as f:
    data = pickle.load(f)
n = data[1].shape[3]   # split=1, last dim = N samples
random.seed(42)
print(" ".join(str(i) for i in random.sample(range(n), min(10, n))))
EOF
)

echo "Validation set sample indices: $INDICES"

for IDX in $INDICES; do
    OUT="repaint_gif_val_logs/repaint_val_sample${IDX}.gif"
    LOG="repaint_gif_val_logs/repaint_val_sample${IDX}.log"
    echo "Running sample $IDX -> $OUT"
    python3.12 repaint_gif.py \
        --split 1 \
        --sample "$IDX" \
        --seed "$IDX" \
        --out "$OUT" \
        "$@" \
        > "$LOG" 2>&1
    echo "  done (log: $LOG)"
done

echo "All 10 validation GIFs saved to $LOGDIR/"

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
