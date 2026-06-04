"""Patches uploaded topo_ddpm files for the server's flat directory layout."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# --- diffusion.py ---
p = os.path.join(BASE, "diffusion.py")
txt = open(p).read()
txt = txt.replace(
    '"Basic DDPM", "diffusion.py"',
    '"diffusion.py"',
)
open(p, "w").write(txt)
print("diffusion.py patched")

# --- train.py ---
p = os.path.join(BASE, "train.py")
lines = open(p).readlines()
lines = [l for l in lines if "_BASIC" not in l]
open(p, "w").writelines(lines)
print("train.py patched")
