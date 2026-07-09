#!/bin/bash
cd /root/DiffusionSummer2026
/root/ocean_ddpm/venv/bin/python "Colored Noise Test/batch_dps_white_vs_annealed.py" \
    --pickle /root/model_pink_noise/data.pickle \
    --ckpt best \
    --zeta 0.04 \
    --out_dir "Colored Noise Test/outputs/dps_white_vs_annealed"
