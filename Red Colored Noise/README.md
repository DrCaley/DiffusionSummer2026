Model: Colored-Gaussian Noise DDPM for 2-D Ocean Currents
=========================================================

Summary
-------
This repository implements a denoising diffusion probabilistic model (DDPM) trained to inpaint and reconstruct 2-D ocean current vector fields (u, v). The model is an epsilon-predicting UNet trained with a cosine noise schedule and colored Gaussian noise (spectral exponent alpha). Inference supports RePaint inpainting and DPS-style guided sampling.

Key components
--------------
- Data loading: `DDPM/dataset.py` — loads `data.pickle` into arrays of shape (N, 2, H, W) and produces a land mask. Land pixels are NaN in the source data and replaced with 0.0 at load time.
- Model: `DDPM/model/unet.py` — UNet that predicts noise ε for input `x_t` and timestep `t`.
- Forward / reconstruction: `DDPM/model/diffusion.py` — `q_sample()` (forward process) and `tweedie_x0()` (posterior mean estimator used to reconstruct `x_0` from `x_t` and ε_pred).
- Noise: `model_parameters/noise_types.py` — `colored_gaussian_noise()` generates Fourier-filtered Gaussian noise with a spectral exponent `alpha` (white/pink/red noise variants).
- Training: `DDPM/train.py` — trains the UNet to predict noise using a masked MSE loss over ocean pixels. The training loop uses the cosine beta schedule and colored noise matching training `noise_alpha`.
- Inference: `DDPM/testing/repaint/repaint_infer.py` — RePaint inpainting sampler. `DDPM/testing/DPS/dps_infer.py` provides an alternative DPS-guided sampler.
- Visualization & evaluation: `DDPM/visualize_infer.py`, `DDPM/visualize_repaint_2x2.py`, and `DDPM/batch_infer.py` generate figures and compute RMSE/MAE on ocean pixels.

How the model is trained
------------------------
- The model predicts noise ε; training minimizes a masked mean-squared error between the model's `eps_pred` and the sampled forward-process noise (ocean pixels only).
- Checkpoints are saved to `DDPM/checkpoints/` (the current best checkpoint is `model_DDPM_MSE_coloredGaussian_cosine.pt`). Training logs are written to `DDPM/checkpoints/training_log.csv`.

Inference / usage examples
--------------------------
Run a single-sample RePaint inference with:

```bash
python DDPM/visualize_infer.py \
  --checkpoint DDPM/checkpoints/model_DDPM_MSE_coloredGaussian_cosine.pt \
  --pickle data.pickle \
  --sample 0 --T 1000 --resample 1 --method repaint --out out.png
```

Batch inference (metrics + images):

```bash
python DDPM/batch_infer.py --checkpoint DDPM/checkpoints/model_DDPM_MSE_coloredGaussian_cosine.pt --pickle data.pickle --out_dir output
```

RePaint sampling is implemented in `DDPM/testing/repaint/repaint_infer.py`. DPS sampling is in `DDPM/testing/DPS/dps_infer.py`.

Evaluation
----------
- Metrics computed: RMSE and MAE on ocean pixels (mask applied) reported in the scripts and plots.
- Visualizations use per-sample `vmax` computed as the 98th percentile of speed to keep color scaling robust to outliers.

Known issue: underestimation of amplitude
----------------------------------------
A consistent observation is that reconstructed vector magnitudes are often smaller than the ground truth. Diagnosis summary:
- The training objective is a masked MSE on predicted noise ε (`model_parameters/loss_functions.py`), so the model effectively learns a posterior mean under MSE. Posterior means are conservative and tend to smooth/attenuate extremes.
- No data rescaling or normalization beyond NaN→0 is applied in `DDPM/dataset.py`, so the issue is not hidden scaling.

Suggested improvements (low-risk first)
--------------------------------------
1. Add an auxiliary magnitude-aware loss to training that reconstructs `x_0` from `eps_pred` (Tweedie formula) and penalizes errors in speed magnitude on ocean pixels.
2. Optionally reweight loss towards larger-magnitude pixels (magnitude-weighted loss) so strong flows contribute more to the objective.
3. Consider predicting `x_0` directly (or adding a second head that predicts `x_0`) if the magnitude bias persists.

Implementation plan for fixing amplitude fidelity
------------------------------------------------
See `IMPLEMENTATION_PLAN.md` for a step-by-step plan that adds a magnitude-aware auxiliary loss, CLI flags, logging, and verification steps.

Reproducibility
---------------
- The repository saves training arguments in checkpoints. To reproduce, use the same checkpoint and `--pickle` file.
- Keep `--noise_alpha` consistent between training and inference to match colored-noise characteristics.

Contact / notes
----------------
If you'd like, I can implement the magnitude-aware auxiliary loss now and run a short training sweep to validate the change.



