import pickle
import numpy as np
from torch.utils.data import Dataset


class PickleFieldDataset(Dataset):
    """Flexible loader for data.pickle. Tries common dict keys.

    Expected shapes:
    - fields: (N, H, W, C) or (N, C, H, W)
    - paths / observations: optional; not required for operator training if full fields exist.
    """
    def __init__(self, pickle_path, split='train', val_fraction=0.1):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Normalize various possible pickle layouts:
        # - dict with array-like entry
        # - single ndarray of shape (N,H,W,C) or (N,C,H,W)
        # - list of ndarrays where each is (H,W,C,Ni) (samples on last axis)
        arr = None
        if isinstance(data, dict):
            candidates = ['fields', 'data', 'X', 'y', 'samples']
            for k in candidates:
                if k in data:
                    arr = data[k]
                    break
            if arr is None:
                # pick first array-like value
                for v in data.values():
                    if hasattr(v, 'shape'):
                        arr = v
                        break
            if arr is None:
                raise RuntimeError('No array-like entry found in pickle')
        else:
            arr = data

        # If list of arrays (e.g., [arr1, arr2, ...]) where each arr has samples on last axis
        if isinstance(arr, list):
            parts = []
            for v in arr:
                a = np.asarray(v)
                if a.ndim == 4:
                    # assume (H,W,C,N) -> move N to axis 0
                    a = np.moveaxis(a, -1, 0)
                    parts.append(a)
                elif a.ndim == 3:
                    # assume (H,W,C) single sample
                    a = a[np.newaxis, ...]
                    parts.append(a)
                else:
                    raise RuntimeError(f'Unsupported array shape in list: {a.shape}')
            arr = np.concatenate(parts, axis=0)
        else:
            arr = np.asarray(arr)

        # Accept both (N,C,H,W) and (N,H,W,C)
        if arr.ndim == 4 and arr.shape[1] in (1,2,3):
            # (N, C, H, W) -> (N, H, W, C)
            arr = np.transpose(arr, (0,2,3,1))

        if arr.ndim != 4:
            raise RuntimeError('Expected array of shape (N,H,W,C) in pickle; got ' + str(arr.shape))

        arr = arr.astype('float32')

        # Replace Inf with NaN so all masking logic is uniform
        arr[np.isinf(arr)] = np.nan

        N = arr.shape[0]
        nan_pct = float(np.isnan(arr).mean()) * 100
        print(f'[dataset] Loaded {N} samples, shape={arr.shape}, NaN%={nan_pct:.1f}')

        # Build persistent land mask: cells that are NaN in ALL samples (land/boundary)
        # Shape: (H, W, C)
        land_mask = np.isnan(arr).all(axis=0)
        print(f'[dataset] Land cells (always-NaN): {int(land_mask.sum())}, ocean: {int((~land_mask).sum())}')

        # Normalize using nanmean/nanstd on training portion (ocean cells only)
        split_idx = int(N * (1 - val_fraction))
        train_arr = arr[:split_idx]
        self.mean = np.nanmean(train_arr, axis=(0, 1, 2), keepdims=True)  # (1,1,1,C)
        self.std  = np.nanstd(train_arr,  axis=(0, 1, 2), keepdims=True).clip(min=1e-8)

        arr = (arr - self.mean) / self.std

        # Fill NaN (land cells + any sporadic missing data) with 0
        arr = np.nan_to_num(arr, nan=0.0)

        # Store land mask for optional use (e.g. masked loss)
        self.land_mask = land_mask  # (H,W,C) bool

        if split == 'train':
            self.arr = arr[:split_idx]
        else:
            self.arr = arr[split_idx:]

    def __len__(self):
        return int(self.arr.shape[0])

    def __getitem__(self, idx):
        return self.arr[idx]
