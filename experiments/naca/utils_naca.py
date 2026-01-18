import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
import os
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import pandas as pd

sys.path.append('../../src')

from deeponet import DeepONet
from noir import OrthogonalLayerDeepONet, FNO2d_layer_ortho
from fno import FNO2d
from wno import WNO2d
from nomad import NOMAD
from utils import DeepONetDataset

def extract_airfoil_boundary(coords_x, coords_y, n_points=256):
    """Extract ordered boundary points from body-fitted mesh."""
    coords = np.stack([coords_x.ravel(), coords_y.ravel()], axis=-1)
    
    # Find airfoil surface (innermost mesh layer)
    distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    surface_threshold = np.percentile(distances, 5)
    surface_mask = distances <= surface_threshold
    surface_points = coords[surface_mask]
    
    # Order by angle
    angles = np.arctan2(surface_points[:, 1], surface_points[:, 0])
    sorted_idx = np.argsort(angles)
    surface_ordered = surface_points[sorted_idx]
    
    # Resample to fixed number of points
    t = np.linspace(0, 1, len(surface_ordered))
    t_new = np.linspace(0, 1, n_points)
    
    fx = interp1d(t, surface_ordered[:, 0], kind='cubic', fill_value='extrapolate')
    fy = interp1d(t, surface_ordered[:, 1], kind='cubic', fill_value='extrapolate')
    
    return np.stack([fx(t_new), fy(t_new)], axis=-1)

def compute_fourier_descriptors(coords_x, coords_y, n_harmonics=64):
    """Compute Fourier descriptors for airfoil geometry."""
    boundary = extract_airfoil_boundary(coords_x, coords_y, n_points=512)
    
    # Convert to complex representation
    z = boundary[:, 0] + 1j * boundary[:, 1]
    z = z - z.mean()  # Center the curve
    
    # Compute FFT
    Z = np.fft.fft(z)
    descriptors = Z[:n_harmonics]
    
    # Return as real-valued vector
    return np.concatenate([descriptors.real, descriptors.imag])

def load_data(config):
    """Load and preprocess NACA data."""
    print("\n📊 Loading NACA data...")
    
    # Load raw data
    data_dir = config['data']['path']
    input_x = np.load(os.path.join(data_dir, 'NACA_Cylinder_X.npy'))
    input_y = np.load(os.path.join(data_dir, 'NACA_Cylinder_Y.npy'))
    output = np.load(os.path.join(data_dir, 'NACA_Cylinder_Q.npy'))[:, 4]  # Mach number
    
    print(f"  Raw data shapes:")
    print(f"    X coords: {input_x.shape}")
    print(f"    Y coords: {input_y.shape}")
    print(f"    Output: {output.shape}")
    
    # Convert to torch tensors
    input_x = torch.tensor(input_x, dtype=torch.float)
    input_y = torch.tensor(input_y, dtype=torch.float)
    input_coords = torch.stack([input_x, input_y], dim=-1)
    output = torch.tensor(output, dtype=torch.float)
    
    # Dataset parameters
    ntrain = config['data']['ntrain']
    ntest = config['data']['ntest']
    batch_size = config['data']['batch_size']
    
    # Spatial resolution
    r1 = r2 = 1
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)
    
    # Split data
    x_train = input_coords[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input_coords[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = output[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
    
    print(f"\n  Data split:")
    print(f"    Training: {x_train.shape}, {y_train.shape}")
    print(f"    Testing: {x_test.shape}, {y_test.shape}")
    
    # Compute Fourier descriptors for branch input
    n_harmonics = config['data']['n_harmonics']
    print(f"\n  Computing Fourier descriptors ({n_harmonics} harmonics)...")
    
    train_branch_fourier = np.array([
        compute_fourier_descriptors(input_x[i], input_y[i], n_harmonics=n_harmonics) 
        for i in range(ntrain)
    ])
    
    test_branch_fourier = np.array([
        compute_fourier_descriptors(input_x[i], input_y[i], n_harmonics=n_harmonics) 
        for i in range(ntrain, ntrain+ntest)
    ])
    
    # Standardize features
    scaler = StandardScaler()
    train_branch = scaler.fit_transform(train_branch_fourier)
    test_branch = scaler.transform(test_branch_fourier)
    
    print(f"    Branch input shape: {train_branch.shape}")
    
    # Create normalized grid for trunk
    grid_x = torch.linspace(0, 1, s1).reshape(s1, 1, 1).repeat(1, s2, 1)
    grid_y = torch.linspace(0, 1, s2).reshape(1, s2, 1).repeat(s1, 1, 1)
    grid = torch.cat([grid_x, grid_y], dim=-1)
    
    # Create data loaders for DeepONet
    train_branch_tensor = torch.tensor(train_branch, dtype=torch.float)
    test_branch_tensor = torch.tensor(test_branch, dtype=torch.float)
    trunk_input = grid.reshape(s1*s2, 2)
    
    train_target_don = y_train.reshape(ntrain, s1*s2)
    test_target_don = y_test.reshape(ntest, s1*s2)
    
    train_loader_don = torch.utils.data.DataLoader(
        DeepONetDataset(train_branch_tensor, trunk_input, train_target_don),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader_don = torch.utils.data.DataLoader(
        DeepONetDataset(test_branch_tensor, trunk_input, test_target_don),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create data loaders for FNO
    train_target_fno = y_train.unsqueeze(-1)
    test_target_fno = y_test.unsqueeze(-1)
    
    train_loader_fno = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, train_target_fno),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader_fno = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, test_target_fno),
        batch_size=batch_size,
        shuffle=False
    )
    
    print("✅ Data loading complete!")
    
    return {
        'train_loader_don': train_loader_don,
        'test_loader_don': test_loader_don,
        'train_loader_fno': train_loader_fno,
        'test_loader_fno': test_loader_fno,
        'scaler': scaler,
        'branch_input_size': train_branch.shape[1],
        's1': s1,
        's2': s2
    }

# ============================================================================
# Model Creation
# ============================================================================

def create_model(model_type, config, branch_input_size):
    """Initialize a model based on type."""
    
    # Get parameters from config
    s1 = config['data']['resolution'][0]
    s2 = config['data']['resolution'][1]
    train_data_res = s1 * s2
    
    if model_type in ['default', 'nomad', 'ortho_layer_ica']:
        width = config['hyperparameters']['width_don']
        HD = config['hyperparameters']['hidden_dim']
    else:
        width = config['hyperparameters']['width_fno'] * 4  # FNO uses width/4 internally
        HD = config['hyperparameters']['hidden_dim']
    
    modes = config['hyperparameters']['modes']
    ica_dim = config['hyperparameters']['ica_dim']
    
    # Architecture definitions
    branch_arch = [branch_input_size, width, width, width, HD]
    trunk_arch = [2, width, width, width]
    activation_fn = nn.ReLU
    
    if model_type == 'default':
        model = DeepONet(
            branch_arch=branch_arch,
            trunk_arch=trunk_arch,
            num_outputs=1,
            activation_fn=activation_fn
        )
    elif model_type == 'nomad':
        combined_input_dim = HD + HD
        combined_arch = [combined_input_dim, width, width, 1]
        model = NOMAD(
            branch_arch=branch_arch,
            trunk_arch=trunk_arch,
            combined_arch=combined_arch,
            num_outputs=1,
            activation_fn=activation_fn
        )
    elif model_type == 'ortho_layer_ica':
        model = OrthogonalLayerDeepONet(
            branch_arch=branch_arch,
            trunk_arch=trunk_arch,
            num_outputs=1,
            activation_fn=activation_fn,
            p_ica=ica_dim,
        )
    elif model_type == 'fourier':
        model = FNO2d(
            modes1=modes[0], 
            modes2=modes[1],
            width=width//4,
            in_channels=2,
            out_channels=1,
        )
    elif model_type == 'fourier_layer_ortho':
        model = FNO2d_layer_ortho(
            modes1=modes[0], 
            modes2=modes[1],
            width=width//4,
            in_channels=2,
            out_channels=1,
            p_ica=ica_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
