#!/usr/bin/env python
"""
Train all NACA models with multiple seeds.
Usage: python train.py [--config config.yaml]
"""
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import time
from utils_naca import load_data, create_model
import pandas as pd

# Add src to path
sys.path.append('../../src')

from losses import LpLoss, ICALoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, test_loader, config, model_type, seed):
    """Train a single model."""
    
    print(f"\n🚀 Training {model_type} with seed {seed}")
    
    model = model.to(device)
    
    # Get training parameters
    epochs = config['training']['epochs']
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    patience = config['training']['patience']
    gamma = config['training']['gamma']
    batch_size = config['data']['batch_size']
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500)
    
    # Loss functions
    criterion = LpLoss(size_average=True)
    orthogonal_loss_fn = ICALoss() if 'ortho' in model_type or 'ica' in model_type else None
    
    # Training state
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    
    # Check if model is FNO-type or DeepONet-type
    fno_types = ['fourier', 'fourier_layer_ortho', 'wavelet']
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
                
                if model_type == 'fourier':
                    output = model(x)
                elif model_type == 'fourier_layer_ortho':
                    output, x_ = model(x)
                    B, H, W, C_out, p_ica = x_.shape
                    x_ = x_.reshape(B * H * W, C_out, p_ica)
                    ortho_loss = orthogonal_loss_fn(x_)
            else:
                x_branch, x_trunk = batch_data[0]
                y = batch_data[1]
                x_branch = x_branch.to(device)
                x_trunk = x_trunk.to(device)
                y = y.unsqueeze(-1).to(device) if y.ndim == 2 else y.to(device)
                
                if model_type in ['default', 'nomad']:
                    output = model(x_branch, x_trunk)
                elif model_type == 'ortho_layer_ica':
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
                    
                    if model_type == 'fourier':
                        output = model(x)
                    elif model_type == 'fourier_layer_ortho':
                        output, _ = model(x)
                else:
                    x_branch, x_trunk = batch_data[0]
                    y = batch_data[1]
                    x_branch = x_branch.to(device)
                    x_trunk = x_trunk.to(device)
                    y = y.unsqueeze(-1).to(device) if y.ndim == 2 else y.to(device)
                    
                    if model_type in ['default', 'nomad']:
                        output = model(x_branch, x_trunk)
                    elif model_type == 'ortho_layer_ica':
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
            if model_type in ['default', 'nomad', 'ortho_layer_ica']:
                width = config['hyperparameters']['width_don']
            else:
                width = config['hyperparameters']['width_fno']
            
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
    print("         NACA AIRFOIL TRAINING")
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
            model = create_model(model_type, config, data_dict['branch_input_size'])
            
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
            model = create_model(model_type, config, data_dict['branch_input_size'])
            
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
