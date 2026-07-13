#!/bin/bash
# Sequential: train VAE first, then train latent DDPM.
set -e
PICKLE="/root/ocean_ddpm/data_local.pickle"
DIR="/root/ldm"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Stage 1: Training VAE ==="
cd /root/autoencoder_train
python3 -u train_vae.py \
    --pickle "$PICKLE" \
    --save_dir "$DIR/checkpoints_vae" \
    --epochs 1000 \
    --batch 32 \
    --beta 0.0001 \
    --c_lat 4 \
    --base_ch 32 \
    --patience 80 \
    > "$DIR/train_vae.log" 2>&1
log "VAE training complete."

log "=== Stage 2: Training Latent DDPM ==="
python3 -u train_latent_ddpm.py \
    --pickle   "$PICKLE" \
    --vae_ckpt "$DIR/checkpoints_vae/best_vae.pt" \
    --save_dir "$DIR/checkpoints_ldm" \
    --epochs   1000 \
    --batch    64 \
    --patience 80 \
    > "$DIR/train_ldm.log" 2>&1
log "Latent DDPM training complete."

log "All done. Checkpoints in $DIR"
