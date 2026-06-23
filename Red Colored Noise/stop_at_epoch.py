"""
stop_at_epoch.py
Watches the training log and kills the training process when it reaches
the target epoch (default 400), unless training has already converged.
"""
import time
import os
import signal

LOG_PATH   = "/root/Colored_Noise_DDPM/DDPM/checkpoints/train.log"
TRAIN_PID  = 73409
STOP_EPOCH = 400


def last_epoch(log_path):
    """Return the last epoch number logged, or 0 if none yet."""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("Epoch"):
                # e.g. "Epoch  117/1500  train=..."
                epoch_str = line.split("/")[0].split()[-1]
                return int(epoch_str)
    except Exception:
        pass
    return 0


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


log(f"Watching training (PID {TRAIN_PID}). Will stop at epoch {STOP_EPOCH}.")

while True:
    if not pid_alive(TRAIN_PID):
        log("Training process is no longer running. Exiting watcher.")
        break

    epoch = last_epoch(LOG_PATH)

    if epoch >= STOP_EPOCH:
        # The train.py log line is written before checkpoints are saved.
        # Wait 120 s to ensure the periodic checkpoint (epoch_400.pt) and
        # best-model checkpoint have fully flushed to disk before stopping.
        log(f"Reached epoch {epoch} >= {STOP_EPOCH}. Waiting 120s for checkpoints to flush...")
        time.sleep(120)

        # Verify checkpoint exists before killing
        ckpt_periodic = f"/root/Colored_Noise_DDPM/DDPM/checkpoints/epoch_{STOP_EPOCH}.pt"
        ckpt_best     = "/root/Colored_Noise_DDPM/DDPM/checkpoints/model_DDPM_MSE_coloredGaussian_cosine.pt"
        for ckpt in (ckpt_periodic, ckpt_best):
            if os.path.isfile(ckpt):
                size_mb = os.path.getsize(ckpt) / 1e6
                log(f"Checkpoint verified: {ckpt}  ({size_mb:.1f} MB)")
            else:
                log(f"WARNING: expected checkpoint not found: {ckpt}")

        log(f"Sending SIGTERM to PID {TRAIN_PID}...")
        try:
            os.kill(TRAIN_PID, signal.SIGTERM)
        except ProcessLookupError:
            log("Process already gone.")
        time.sleep(15)
        if pid_alive(TRAIN_PID):
            log("Process still alive after SIGTERM, sending SIGKILL...")
            try:
                os.kill(TRAIN_PID, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log("Training stopped. Checkpoints ready for resuming.")
        break

    log(f"Epoch {epoch}/{STOP_EPOCH} — still training...")
    time.sleep(60)
