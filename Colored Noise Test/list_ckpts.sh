#!/bin/bash
BASE="/root/DiffusionSummer2026/Colored Noise Test"
for d in white_noise pink_noise red_noise pink_noise_full red_noise_full annealed_noise; do
    echo "=== $d ==="
    ls "$BASE/$d/checkpoints/" 2>/dev/null
done
