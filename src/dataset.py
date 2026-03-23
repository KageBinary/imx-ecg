import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import PROCESSED_DIR


class ECGDataset(Dataset):

    def __init__(self, split):
        """
        split must be:
        'train', 'val', or 'test'
        """

        x_path = os.path.join(PROCESSED_DIR, f"X_{split}.npy")
        y_path = os.path.join(PROCESSED_DIR, f"y_{split}.npy")

        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        signal = self.X[idx]
        label = self.y[idx]

        # Add channel dimension for Conv1D
        signal = np.expand_dims(signal, axis=0)

        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return signal, label