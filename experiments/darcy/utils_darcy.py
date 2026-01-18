import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from scipy import io

# Add src to path
sys.path.append('../../src')

from deeponet import DeepONet
from noir import OrthogonalLayerDeepONet, FNO2d_layer_ortho
from fno import FNO2d
from wno import WNO2d
from nomad import NOMAD
from utils import DeepONetDataset

class MatReader:
    """Utility for reading .mat files."""
    
    def __init__(self, file_path):
        self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load .mat file."""
        self.data = io.loadmat(file_path)
    
    def read_field(self, field):
        """Read specific field from .mat file."""
        x = self.data[field]
        return x if isinstance(x, np.ndarray) else x.toarray()

def load_data(config):
    """Load and preprocess Darcy data."""
    print("\n📊 Loading Darcy data...")
    
    # Parameters
    ntrain = config['data']['ntrain']
    ntest = config['data']['ntest']
    batch_size = config['data']['batch_size']
    s = config['data']['resolution']
    r = config['data']['downsample_rate']
    
    train_data_res = s ** 2
    
    # Load training data
    print(f"  Loading training data from: {config['data']['train_path']}")
    reader = MatReader(config['data']['train_path'])
    x_train_fno = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
    y_train_fno = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]
    
    # Load test data
    print(f"  Loading test data from: {config['data']['test_path']}")
    reader.load_file(config['data']['test_path'])
    x_test_fno = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
    y_test_fno = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]
    
    print(f"\n  Data shapes:")
    print(f"    Training input: {x_train_fno.shape}")
    print(f"    Training output: {y_train_fno.shape}")
    print(f"    Test input: {x_test_fno.shape}")
    print(f"    Test output: {y_test_fno.shape}")
    
    # Create grid for coordinates
    grid_1d = np.linspace(0, 1, s)
    xx, yy = np.meshgrid(grid_1d, grid_1d)
    grid = np.stack([xx, yy], axis=-1)
    grid = grid.reshape(1, s, s, 2)
    grid = torch.tensor(grid, dtype=torch.float)
    
    # Prepare DeepONet format data
    x_train_don = (x_train_fno.reshape(ntrain, s*s), grid.reshape(s*s, 2))
    y_train_don = y_train_fno.reshape(ntrain, s*s)
    x_test_don = (x_test_fno.reshape(ntest, s*s), grid.reshape(s*s, 2))
    y_test_don = y_test_fno.reshape(ntest, s*s)
    
    # Create data loaders for DeepONet
    branch_input_train = torch.tensor(x_train_don[0], dtype=torch.float)
    trunk_input = torch.tensor(x_train_don[1], dtype=torch.float)
    train_target_don = torch.tensor(y_train_don, dtype=torch.float)
    
    branch_input_test = torch.tensor(x_test_don[0], dtype=torch.float)
    test_target_don = torch.tensor(y_test_don, dtype=torch.float)
    
    train_loader_don = torch.utils.data.DataLoader(
        DeepONetDataset(branch_input_train, trunk_input, train_target_don),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader_don = torch.utils.data.DataLoader(
        DeepONetDataset(branch_input_test, trunk_input, test_target_don),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create data loaders for FNO
    train_input_fno = torch.tensor(x_train_fno, dtype=torch.float)
    train_target_fno = torch.tensor(y_train_fno, dtype=torch.float)
    test_input_fno = torch.tensor(x_test_fno, dtype=torch.float)
    test_target_fno = torch.tensor(y_test_fno, dtype=torch.float)
    
    # Add grid to FNO input
    train_input_fno = torch.cat([
        train_input_fno.reshape(ntrain, s, s, 1),
        grid.repeat(ntrain, 1, 1, 1)
    ], dim=3)
    
    test_input_fno = torch.cat([
        test_input_fno.reshape(ntest, s, s, 1),
        grid.repeat(ntest, 1, 1, 1)
    ], dim=3)
    
    train_loader_fno = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_input_fno, train_target_fno),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader_fno = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_input_fno, test_target_fno),
        batch_size=batch_size,
        shuffle=False
    )
    
    print("✅ Data loading complete!")
    
    return {
        'train_loader_don': train_loader_don,
        'test_loader_don': test_loader_don,
        'train_loader_fno': train_loader_fno,
        'test_loader_fno': test_loader_fno,
        'train_data_res': train_data_res,
        's': s,
        'grid': grid
    }

# ============================================================================
# Model Creation
# ============================================================================

def create_model(model_type, config, train_data_res):
    """Initialize a model based on type."""
    
    # Get parameters from config
    s = config['data']['resolution']
    
    if model_type in ['default', 'nomad', 'ortho_layer_ica']:
        width = config['hyperparameters']['width_don']
        HD = config['hyperparameters']['hidden_dim']
    else:
        width = config['hyperparameters']['width_fno'] * 4  # FNO uses width/4 internally
        HD = config['hyperparameters']['hidden_dim']
    
    modes = config['hyperparameters']['modes']
    ica_dim = config['hyperparameters']['ica_dim']
    
    # Architecture definitions
    N_func = 1
    N_coord = 2
    N_outputs = 1
    
    branch_arch = [train_data_res * N_func, width, width, width, HD]
    trunk_arch = [N_coord, width, width, width]
    activation_fn = nn.ReLU
    
    if model_type == 'default':
        model = DeepONet(
            branch_arch=branch_arch,
            trunk_arch=trunk_arch,
            num_outputs=N_outputs,
            activation_fn=activation_fn
        )
    elif model_type == 'nomad':
        combined_input_dim = HD + (HD * N_outputs)
        combined_arch = [combined_input_dim, width, width, N_outputs]
        model = NOMAD(
            branch_arch=branch_arch,
            trunk_arch=trunk_arch,
            combined_arch=combined_arch,
            num_outputs=N_outputs,
            activation_fn=activation_fn
        )
    elif model_type == 'ortho_layer_ica':
        model = OrthogonalLayerDeepONet(
            branch_arch=branch_arch,
            trunk_arch=trunk_arch,
            num_outputs=N_outputs,
            activation_fn=activation_fn,
            p_ica=ica_dim,
        )
    elif model_type == 'fourier':
        model = FNO2d(
            modes1=modes,
            modes2=modes,
            width=width//4,
            in_channels=3,  # Input + 2D coordinates
            out_channels=N_outputs
        )
    elif model_type == 'fourier_layer_ortho':
        model = FNO2d_layer_ortho(
            modes1=modes,
            modes2=modes,
            width=width//4,
            in_channels=3,
            out_channels=N_outputs,
            p_ica=ica_dim,
        )
    elif model_type == 'wavelet':
        model = WNO2d(
            width=width//4,
            level=5,
            layers=3,
            size=[s, s],
            wavelet=['near_sym_b', 'qshift_b'],
            in_channel=3,
            out_channel=N_outputs,
            grid_range=[1, 1],
            omega=8,
            padding=1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model