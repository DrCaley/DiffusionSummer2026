#!/bin/bash
set -e
TRAIN="/root/ocean_diffusion/Conditional DDPM/train.py"
LOGDIR="/root/ocean_diffusion/Conditional DDPM"
PICKLE="/root/ocean_diffusion/data.pickle"

# Run all three sequentially in a single tmux session so they don't compete for GPU
tmux new-session -d -s cond_train -x 220 -y 50
tmux send-keys -t cond_train "
echo '=== Starting voronoi ===' | tee \"$LOGDIR/train_voronoi.log\" &&
python3 \"$TRAIN\" --cond voronoi --epochs 400 --pickle \"$PICKLE\" 2>&1 | tee -a \"$LOGDIR/train_voronoi.log\" &&
echo '=== Starting path ===' | tee \"$LOGDIR/train_path.log\" &&
python3 \"$TRAIN\" --cond path   --epochs 400 --pickle \"$PICKLE\" 2>&1 | tee -a \"$LOGDIR/train_path.log\" &&
echo '=== Starting both ===' | tee \"$LOGDIR/train_both.log\" &&
python3 \"$TRAIN\" --cond both   --epochs 400 --pickle \"$PICKLE\" 2>&1 | tee -a \"$LOGDIR/train_both.log\" &&
echo '=== All three done ==='
" Enter

tmux ls
