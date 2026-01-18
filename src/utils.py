import numpy as np

def relative_l2_error(y_true, y_pred):
    """Calculates mean relative L2 error over a batch."""
    if y_true.ndim > 2 or y_pred.ndim > 2:
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        
    y_true_norm = np.linalg.norm(y_true, ord=2, axis=1)
    diff_norm = np.linalg.norm(y_true - y_pred, ord=2, axis=1)
    
    mask = y_true_norm > 1e-8
    
    if np.any(mask):
        return np.mean(diff_norm[mask] / y_true_norm[mask])
    else:
        return 0.0

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import urllib.request
from scipy import io


class DeepONetDataset(Dataset):
    ''' 
    Custom dataset class for the DeepONet model.
    
    Args:
    - branch_data: Branch input data, shape (num_samples, input_size)
    - trunk_data: Trunk input data, shape (num_trunk_points, trunk_size)
    - target_data: Target output data, shape (num_samples, num_trunk_points)
    
    This dataset assumes the trunk input is shared across all samples.
    '''
    
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)    # Shared trunk input (100, 1)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.branch_data)  # Return the number of samples

    def __getitem__(self, idx):
        # Get the branch input and target output for this index
        branch_input = self.branch_data[idx]
        target_output = self.target_data[idx]
        # Return the branch input, shared trunk input (same for all samples), and target output
        return (branch_input, self.trunk_data), target_output
