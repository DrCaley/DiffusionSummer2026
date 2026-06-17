#!/bin/bash
cd /root/ocean_diffusion
python3 "Model Parameters/Loss Function/batch_eval_multi_run.py" \
    --ckpt_dir "Model Parameters/Loss Function/all_models" \
    --n_samples 50 --n_runs 10 --path_steps 150 --resample 10 --seed 42 \
    > "Model Parameters/Loss Function/multirun_eval.log" 2>&1
