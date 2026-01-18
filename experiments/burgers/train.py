#!/usr/bin/env python
"""
Train all Burgers equation models with multiple seeds.
Usage: python train.py [--config config.yaml]
"""

import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Add src to path
sys.path.append('../../src')

from deeponet import DeepONet
from noir import OrthogonalLayerDeepONet, FNO1d_layer_ortho
from fno import FNO1d
from wno import WNO1d
from nomad import NOMAD
from losses import LpLoss, ICALoss
from utils import DeepONetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Data Processing Functions
# ============================================================================

def load_data(config):
    """Load and preprocess Burgers equation data."""
    print("\n📊 Loading Burgers data...")
    
    # Load data
    data_path = config['data']['data_path']
    data = torch.load(data_path)
    
    ntrain = config['data']['ntrain']
    ntest = config['data']['ntest']
    batch_size = config['data']['batch_size']
    s = config['data']['resolution']
    scaling = config['data']['scaling']
    
    # Extract data arrays
    x_train = data['u0_train'].numpy()
    y_train = data['uf_train'].numpy()
    x_test = data['u0_test'].numpy()
    y_test = data['uf_test'].numpy()
    
    print(f"  Raw data shapes:")
    print(f"    Training input: {x_train.shape}")
    print(f"    Training output: {y_train.shape}")
    print(f"    Test input: {x_test.shape}")
    print(f"    Test output: {y_test.shape}")
    
    # Apply scaling if requested
    if scaling:
        print("  Applying MinMaxScaler...")
        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()
        
        train_input = scaler.fit_transform(x_train)
        test_input = scaler.transform(x_test)
        train_target = scaler_target.fit_transform(y_train)
        test_target = scaler_target.transform(y_test)
    else:
        train_input = x_train
        test_input = x_test
        train_target = y_train
        test_target = y_test
    
    # Convert to tensors
    train_input = torch.tensor(train_input, dtype=torch.float)
    train_target = torch.tensor(train_target, dtype=torch.float)
    test_input = torch.tensor(test_input, dtype=torch.float)
    test_target = torch.tensor(test_target, dtype=torch.float)
    
    # Create spatial grid
    grid = data['x_grid'].numpy()
    grid = torch.tensor(grid, dtype=torch.float)
    trunk_input = grid.reshape(s, 1)
    
    # Create DeepONet data loaders
    train_loader_don = DeepONetDataset(
        train_input.unsqueeze(-1), 
        trunk_input, 
        train_target.reshape(ntrain, s, 1)
    )
    train_loader_don = torch.utils.data.DataLoader(
        train_loader_don, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader_don = DeepONetDataset(
        test_input.unsqueeze(-1), 
        trunk_input, 
        test_target.reshape(ntest, s, 1)
    )
    test_loader_don = torch.utils.data.DataLoader(
        test_loader_don, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Create FNO data loaders
    train_loader_fno = torch.utils.data.TensorDataset(
        torch.cat([
            train_input.unsqueeze(-1), 
            trunk_input.unsqueeze(0).repeat(ntrain, 1, 1)
        ], dim=-1),
        train_target.unsqueeze(-1)
    )
    train_loader_fno = torch.utils.data.DataLoader(
        train_loader_fno, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader_fno = torch.utils.data.TensorDataset(
        torch.cat([
            test_input.unsqueeze(-1), 
            trunk_input.unsqueeze(0).repeat(ntest, 1, 1)
        ], dim=-1),
        test_target.unsqueeze(-1)
    )
    test_loader_fno = torch.utils.data.DataLoader(
        test_loader_fno, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print("✅ Data loading complete!")
    
    return {
        'train_loader_don': train_loader_don,
        'test_loader_don': test_loader_don,
        'train_loader_fno': train_loader_fno,
        'test_loader_fno': test_loader_fno,
        'train_input': train_input,
        'test_input': test_input,
        'train_target': train_target,
        'test_target': test_target,
        'grid': grid,
        's': s,
        'scaler': scaler if scaling else None,
        'scaler_target': scaler_target if scaling else None
    }

# ============================================================================
# Model Creation
# ============================================================================

def create_model(model_type, config):
    """Initialize a model based on type."""
    
    s = config['data']['resolution']
    
    if model_type in ['default', 'nomad', 'ortho_ica']:
        width = config['hyperparameters']['width_don']
        HD = config['hyperparameters']['hidden_dim']
    else:
        width = config['hyperparameters']['width_fno'] * 2  # FNO1d uses width/2 internally
        HD = config['hyperparameters']['hidden_dim']
    
    modes = config['hyperparameters']['modes']
    ica_dim = config['hyperparameters']['ica_dim']
    
    # Architecture definitions
    N_outputs = 1
    N_coord = 1
    N_func = 1
    
    branch_arch = [s * N_func, width, width, width, HD]
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
    elif model_type == 'ortho_ica':
        model = OrthogonalLayerDeepONet(
            branch_arch=branch_arch,
            trunk_arch=trunk_arch,
            num_outputs=N_outputs,
            activation_fn=activation_fn,
            p_ica=ica_dim,
        )
    elif model_type == 'fourier':
        model = FNO1d(
            modes=modes,
            width=width//2
        )
    elif model_type == 'fourier_ortho':
        model = FNO1d_layer_ortho(
            modes=modes,
            width=width//2,
            p_ica=ica_dim,
        )
    elif model_type == 'wavelet':
        model = WNO1d(
            width=width//2,
            level=5,
            layers=4,
            size=s,
            wavelet='db2',
            in_channel=2,  # Input + coordinate
            grid_range=1,
            omega=4,
            padding=0
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, test_loader, config, model_type, seed):
    """Train a single model."""
    
    print(f"\n🚀 Training {model_type} with seed {seed}")
    
    model = model.to(device)
    
    # Get training parameters
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    patience = config['training']['patience']
    gamma = config['training']['gamma']
    scheduler_step = config['training']['scheduler_step']
    scheduler_gamma = config['training']['scheduler_gamma']
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_step, 
        gamma=scheduler_gamma
    )
    
    # Loss functions
    criterion = LpLoss(size_average=True)
    orthogonal_loss_fn = ICALoss() if 'ortho' in model_type or 'ica' in model_type else None
    
    # Training state
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    
    # Check if model is FNO-type or DeepONet-type
    fno_types = ['fourier', 'fourier_ortho', 'wavelet']
    is_fno = model_type in fno_types
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_ortho_loss = 0.0
        
        for batch_data in train_loader:
            optimizer.zero_grad()
            ortho_loss = 0.0
            
            if is_fno:
                x, y = batch_data
                x, y = x.to(device), y.to(device)
                
                if model_type == 'fourier' or model_type == 'wavelet':
                    output = model(x)
                elif model_type == 'fourier_ortho':
                    output, x_ = model(x)
                    B, W, C_out, p_ica = x_.shape
                    x_ = x_.reshape(B * W, C_out, p_ica)
                    ortho_loss = orthogonal_loss_fn(x_)
            else:
                x_branch, x_trunk = batch_data[0]
                y = batch_data[1]
                x_branch = x_branch.to(device)
                x_trunk = x_trunk.to(device)
                y = y.to(device)
                
                if model_type in ['default', 'nomad']:
                    output = model(x_branch, x_trunk)
                elif model_type == 'ortho_ica':
                    output, x_ = model(x_branch, x_trunk)
                    B, P, C, d = x_.shape
                    x_ = x_.reshape(B * P, C, d)
                    ortho_loss = orthogonal_loss_fn(x_)
            
            loss_mse = criterion(output, y)
            loss = loss_mse + gamma * ortho_loss if isinstance(ortho_loss, torch.Tensor) else loss_mse
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss_mse.item() * y.size(0)
            if isinstance(ortho_loss, torch.Tensor):
                train_ortho_loss += ortho_loss.item() * y.size(0)
        
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_ortho_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in test_loader:
                if is_fno:
                    x, y = batch_data
                    x, y = x.to(device), y.to(device)
                    
                    if model_type == 'fourier' or model_type == 'wavelet':
                        output = model(x)
                    elif model_type == 'fourier_ortho':
                        output, _ = model(x)
                else:
                    x_branch, x_trunk = batch_data[0]
                    y = batch_data[1]
                    x_branch = x_branch.to(device)
                    x_trunk = x_trunk.to(device)
                    y = y.to(device)
                    
                    if model_type in ['default', 'nomad']:
                        output = model(x_branch, x_trunk)
                    elif model_type == 'ortho_ica':
                        output, _ = model(x_branch, x_trunk)
                
                loss = criterion(output, y)
                val_loss += loss.item() * y.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        # Print progress
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:4d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Ortho: {train_ortho_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            # Save best model
            save_path = Path(config['paths']['save_models'])
            save_path.mkdir(parents=True, exist_ok=True)
            
            model_file = save_path / f"best_model_{model_type}_seed{seed}.pth"
            torch.save(model.state_dict(), model_file)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Save training history
    log_path = Path(config['paths']['save_logs'])
    log_path.mkdir(parents=True, exist_ok=True)
    
    np.save(log_path / f"train_loss_{model_type}_seed{seed}.npy", np.array(train_losses))
    np.save(log_path / f"val_loss_{model_type}_seed{seed}.npy", np.array(val_losses))
    
    # Log summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Training complete!")
    print(f"     Parameters: {total_params:,}")
    print(f"     Time: {training_time:.1f}s")
    print(f"     Best val loss: {best_val_loss:.6f}")
    
    return model, best_val_loss

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("         BURGERS EQUATION TRAINING")
    print("="*70)
    
    # Load data
    data_dict = load_data(config)
    
    # Get model types
    model_types_don = config['models']['types_don']
    model_types_fno = config['models']['types_fno']
    seeds = config['training']['seeds']
    
    # Summary storage
    summary_data = []
    
    # Train DeepONet-based models
    print("\n" + "="*70)
    print("TRAINING DEEPONET-BASED MODELS")
    print("="*70)
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for model_type in model_types_don:
            model = create_model(model_type, config)
            
            trained_model, best_loss = train_model(
                model,
                data_dict['train_loader_don'],
                data_dict['test_loader_don'],
                config,
                model_type,
                seed
            )
            
            summary_data.append({
                'model_type': model_type,
                'seed': seed,
                'best_val_loss': best_loss,
                'params': sum(p.numel() for p in model.parameters())
            })
    
    # Train FNO-based models
    print("\n" + "="*70)
    print("TRAINING FNO-BASED MODELS")
    print("="*70)
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for model_type in model_types_fno:
            model = create_model(model_type, config)
            
            trained_model, best_loss = train_model(
                model,
                data_dict['train_loader_fno'],
                data_dict['test_loader_fno'],
                config,
                model_type,
                seed
            )
            
            summary_data.append({
                'model_type': model_type,
                'seed': seed,
                'best_val_loss': best_loss,
                'params': sum(p.numel() for p in model.parameters())
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(config['paths']['save_logs']) / 'training_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*70)
    print("🎉 ALL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved to: {config['paths']['save_models']}")
    print(f"Logs saved to: {config['paths']['save_logs']}")
    print(f"\nSummary:\n{summary_df.groupby('model_type')['best_val_loss'].agg(['mean', 'std'])}")

if __name__ == '__main__':
    main()