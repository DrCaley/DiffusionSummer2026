Resume bundle for the True Divergent Free Noise run.

Contains:
- checkpoints/best_model.pt
- checkpoints/final_model.pt
- checkpoints/training_log.csv
- data_divfree.pickle
- source code in Basic DDPM/

To continue training later, run from Basic DDPM/:
python train.py --pickle ../data_divfree.pickle --epochs 400 --batch 32 --save-every 100 --rebuild-clean-data
