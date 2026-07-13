Neural Operator training (Fourier Neural Operator)

Files:
- `model_fno.py`: simple FNO2d implementation (PyTorch)
- `dataset.py`: flexible loader for `data.pickle` (expects a dict with arrays)
- `train_fno.py`: training script with validation and early stopping (patience arg)
- `requirements.txt`: minimal dependencies

Usage example:
```
python train_fno.py --data-path ../ocean_ddpm/data.pickle --out-dir ./checkpoints --epochs 500 --patience 50
```
