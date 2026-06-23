#!/bin/bash
cd /root/ocean_diffusion
python3.12 "Conditional DDPM/infer.py" --cond voronoi --pickle data.pickle > "Conditional DDPM/infer_voronoi.log" 2>&1
python3.12 "Conditional DDPM/infer.py" --cond path    --pickle data.pickle > "Conditional DDPM/infer_path.log"    2>&1
python3.12 "Conditional DDPM/infer.py" --cond both    --pickle data.pickle > "Conditional DDPM/infer_both.log"    2>&1
echo ALL_DONE
