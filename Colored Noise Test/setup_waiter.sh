#!/bin/bash
# Get the annealed training PID
APID=$(pgrep -f 'annealed_noise/train.py' | head -1)
echo "Annealed PID: $APID"
if [ -z "$APID" ]; then
    echo "ERROR: annealed training not found!"
    exit 1
fi
# Send commands to the waiter tmux window
tmux send-keys -t ssh_tmux:waiter "source /root/ocean_ddpm/venv/bin/activate && cd /root/DiffusionSummer2026" Enter
sleep 1
tmux send-keys -t ssh_tmux:waiter "bash 'Colored Noise Test/wait_then_train_full_200.sh' $APID 2>&1 | tee 'Colored Noise Test/waiter.log'" Enter
echo "Waiter launched for PID $APID"
