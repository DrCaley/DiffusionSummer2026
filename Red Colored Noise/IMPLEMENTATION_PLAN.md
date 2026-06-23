# Implementation Plan: Improve amplitude fidelity

## Goal
Improve the model so reconstructed current magnitudes are closer to ground truth, while keeping the existing DDPM/RePaint pipeline mostly intact.

## Plan
1. Add a magnitude-aware auxiliary loss in `DDPM/train.py`.
2. Reconstruct `x_0` from `eps_pred` during training using the existing Tweedie formula.
3. Compare predicted vs true speed magnitude on ocean pixels.
4. Keep the current masked epsilon MSE as the main loss, and blend in the new term with a small tunable weight.
5. Add CLI flags and checkpoint metadata for the new loss weights.
6. Log the new loss component during training and validation.
7. Extend `DDPM/batch_infer.py` to report magnitude RMSE for before/after comparison.
8. Do a visual spot-check in `DDPM/visualize_infer.py` to confirm stronger currents are less compressed.

## Relevant files
- `DDPM/train.py`
- `model_parameters/loss_functions.py`
- `DDPM/model/diffusion.py`
- `DDPM/batch_infer.py`
- `DDPM/visualize_infer.py`

## Verification
- Run a short training pass and confirm the new loss is finite.
- Check that validation still decreases.
- Compare baseline vs modified model on the same evaluation set using magnitude RMSE.
- Confirm land pixels remain excluded from both loss and metrics.
