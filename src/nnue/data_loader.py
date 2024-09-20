import chess 
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file_path, num_features):
        self.file_path = file_path
        self.num_features = num_features
        with h5py.File(self.file_path, 'r') as f:
            self.len = f['X_b'].shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            black_ones = f['X_b'][idx]
            white_ones = f['X_w'][idx]
            evaluation = f['y'][idx]
        evaluation = torch.tensor(evaluation, dtype=torch.float32)

        black_features = torch.zeros(self.num_features, dtype=torch.float32)
        white_features = torch.zeros(self.num_features, dtype=torch.float32)
        
        black_features[black_ones] = 1.0
        white_features[white_ones] = 1.0

        return white_features, black_features,  evaluation
