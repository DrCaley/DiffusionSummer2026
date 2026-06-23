"""
wait_and_train.py
Waits for the Divergence_Free_DDPM training process (PID 37963) to finish,
then launches the Colored Gaussian Noise DDPM training.
"""
import os
import subprocess
import sys
import time

EXISTING_PID = 37963
TRAIN_DIR    = "/root/Colored_Noise_DDPM/DDPM"
PICKLE       = "../../data.pickle"
CKPT_DIR     = "checkpoints"
LOG_PATH     = "/root/Colored_Noise_DDPM/DDPM/checkpoints/train.log"


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


os.makedirs(os.path.join(TRAIN_DIR, CKPT_DIR), exist_ok=True)

log(f"Waiting for PID {EXISTING_PID} (Divergence_Free_DDPM) to finish...")
while pid_alive(EXISTING_PID):
    time.sleep(120)
    log(f"Still waiting for PID {EXISTING_PID} ...")

log(f"PID {EXISTING_PID} has finished. Launching Colored Gaussian Noise DDPM training...")

cmd = [
    sys.executable, "train.py",
    "--pickle", PICKLE,
    "--epochs", "1500",
    "--batch", "32",
    "--workers", "4",
]

with open(LOG_PATH, "w") as log_fh:
    proc = subprocess.Popen(
        cmd,
        cwd=TRAIN_DIR,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

pid_file = "/root/Colored_Noise_DDPM/train_pid.txt"
with open(pid_file, "w") as f:
    f.write(str(proc.pid) + "\n")

log(f"Training started with PID={proc.pid}. Log: {LOG_PATH}")
