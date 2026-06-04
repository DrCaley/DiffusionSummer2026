"""Patches uploaded multi_loss_ddpm files for the server's flat directory layout."""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

# diffusion.py: change  "..", "Basic DDPM", "diffusion.py"  →  "..", "diffusion.py"
p = os.path.join(BASE, "diffusion.py")
txt = open(p).read()
txt = txt.replace(
    'os.path.join(\n    os.path.dirname(__file__), "..", "Basic DDPM", "diffusion.py"\n)',
    'os.path.join(\n    os.path.dirname(__file__), "..", "diffusion.py"\n)',
)
open(p, "w").write(txt)
print("diffusion.py patched")

# train.py: remove the line that appends Basic DDPM to sys.path
# (model.py lives in the root on the server)
p = os.path.join(BASE, "train.py")
lines = open(p).readlines()
lines = [l for l in lines if '"Basic DDPM"' not in l]
open(p, "w").writelines(lines)
print("train.py patched")

# Install geomloss if not already present (needed for --loss wasserstein)
try:
    import geomloss  # noqa: F401
    print("geomloss already installed")
except ImportError:
    print("Installing geomloss...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "geomloss"])
    print("geomloss installed")
