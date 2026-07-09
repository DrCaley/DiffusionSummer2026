import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class OceanCurrentDataset(Dataset):
    """
    Loads ocean current vector fields from a tidal pickle.

    Expects dict-format pickle with keys:
        "train" / "val" / "test" : (H, W, 2, N) float32
        "features" : {"train": (N, feat_dim), ...} float32 precomputed features

    __getitem__ returns (x, feat):
        x    : (2, H, W) float32 tensor
        feat : (feat_dim,) float32 tensor
    """

    def __init__(self, pickle_path: str, split: int = 0):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        key = ("train", "val", "test")[split]
        arr = data[key]                          # (H, W, 2, N)
        H, W, _, N = arr.shape

        arr_t = np.transpose(arr, (3, 2, 0, 1)).astype(np.float32)  # (N, 2, H, W)

        self.land_mask = torch.from_numpy(np.isnan(arr[:, :, 0, 0]))  # (H, W)
        self.data      = torch.from_numpy(np.nan_to_num(arr_t, nan=0.0))

        # Precomputed features
        feats = data["features"][key]            # (N, feat_dim)
        self.features   = torch.from_numpy(feats.astype(np.float32))
        self.feat_dim   = feats.shape[1]
        self.feat_names = data.get("feature_names", [f"f{i}" for i in range(self.feat_dim)])

    @staticmethod
    def compute_stats(pickle_path: str, split: int = 0):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        key = ("train", "val", "test")[split]
        arr = data[key]
        ocean_mask = ~np.isnan(arr[:, :, 0, 0])
        vals = arr[ocean_mask, :, :].flatten()
        vals = vals[~np.isnan(vals)].astype(np.float64)
        return float(vals.mean()), float(vals.std())

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return self.data[idx], self.features[idx]
