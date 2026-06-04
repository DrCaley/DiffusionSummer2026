import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class OceanCurrentDataset(Dataset):
    """
    Loads ocean current vector fields from the pickle file.
    Each sample is a (2, H, W) float32 tensor — channels are u and v.
    Land pixels (originally NaN) are set to 0.
    The land_mask attribute is a (H, W) bool tensor: True = land.
    """

    def __init__(self, pickle_path: str, split: int = 0):
        """
        Args:
            pickle_path: path to data.pickle
            split: 0 = train, 1 = val, 2 = test
        """
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        arr = data[split]  # (H, W, 2, N)
        H, W, _, N = arr.shape

        # Rearrange to (N, 2, H, W)
        arr_t = np.transpose(arr, (3, 2, 0, 1)).astype(np.float32)

        # Land mask: True where NaN (same for all timesteps)
        self.land_mask = torch.from_numpy(np.isnan(arr[:, :, 0, 0]))  # (H, W)

        # Replace NaN with 0 so tensors are finite
        self.data = torch.from_numpy(np.nan_to_num(arr_t, nan=0.0))  # (N, 2, H, W)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]  # (2, H, W)
