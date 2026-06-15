# Journal

## Initial setup

- Read the project instructions and data description.
- Confirmed the workspace initially contained only `ModelInstructions.txt` and `data.pickle`.
- Chose a fresh PyTorch project layout under `Basic DDPM/` so the documented quick-start commands remain usable.

## Implementation plan

- Clean the source pickle into a divergence-free version while keeping the land mask intact as `NaN`.
- Train a cosine-schedule DDPM with epsilon prediction and masked MSE over ocean pixels.
- Use divergence-free Gaussian noise for both forward diffusion corruption and reverse sampling.
- Add RePaint-style conditioning with sparse robot path observations.
- Save checkpoints every 100 epochs and keep the best validation checkpoint.
- Write a CSV training log with epoch, UTC timestamp, train loss, and validation loss so the curve can be graphed later.
- Generate separate actual, predicted, and loss figures with vector arrows and the robot path overlay.

## Status

- Project scaffold and core modules added.
- Validation and remote transfer still need to be run.
- Training now records per-epoch losses in `checkpoints/training_log.csv`.
