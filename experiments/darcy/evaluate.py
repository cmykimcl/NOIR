#!/usr/bin/env python
"""
Evaluate noise robustness of trained Darcy models.
Usage: python evaluate.py [--config config.yaml]
"""

import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append('../../src')
sys.path.append('.')

from noise_generators import NoiseGeneratorDarcy
from train import load_data, create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_with_noise(model, test_loader, noise_type, noise_level, noise_fraction, 
                       config, model_type, input_data_std):
    """Evaluate model performance with specific noise configuration."""
    
    model.eval()
    
    # Check if model is FNO-type or DeepONet-type
    fno_types = ['fourier', 'fourier_layer_ortho', 'wavelet']
    is_fno = model_type in fno_types
    
    s = config['data']['resolution']
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if is_fno:
                x, y = batch_data
                x_func = x[:, :, :, 0]  # Permeability field
                x_coord = x[:, :, :, 1:].to(device)  # Grid coordinates
                y = y.to(device)
                
                # Apply noise to permeability field
                batch_size = x_func.shape[0]
                generator_shape = (batch_size, s, s)
                noise_gen = NoiseGeneratorDarcy(x_shape=generator_shape, device=device)
                
                if noise_level > 0:
                    noise_std = noise_level * input_data_std
                    x_noisy, _ = noise_gen.generate(
                        x=x_func.to(device),
                        noise_type=noise_type,
                        noise_std=noise_std,
                        k_fraction=noise_fraction
                    )
                else:
                    x_noisy = x_func.to(device)
                
                # Reconstruct input
                x_noisy_full = torch.cat([x_noisy.unsqueeze(-1), x_coord], dim=-1)
                
                # Get prediction
                if model_type in ['fourier', 'wavelet']:
                    output = model(x_noisy_full)
                    output = output.squeeze(-1)
                else:  # fourier_layer_ortho
                    output, _ = model(x_noisy_full)
                    output = output.squeeze(-1)
                
            else:  # DeepONet models
                x_branch, x_trunk = batch_data[0]
                y = batch_data[1]
                x_trunk = x_trunk.to(device)
                y = y.to(device)
                
                # Apply noise to branch input
                batch_size = x_branch.shape[0]
                generator_shape = (batch_size, s, s)
                noise_gen = NoiseGeneratorDarcy(x_shape=generator_shape, device=device)
                
                if noise_level > 0:
                    noise_std = noise_level * input_data_std
                    x_branch_3d, _ = noise_gen.generate(
                        x=x_branch.reshape(batch_size, s, s).to(device),
                        noise_type=noise_type,
                        noise_std=noise_std,
                        k_fraction=noise_fraction
                    )
                    x_branch_noisy = x_branch_3d.reshape(batch_size, -1)
                else:
                    x_branch_noisy = x_branch.to(device)
                
                # Get prediction
                if model_type in ['default', 'nomad']:
                    output = model(x_branch_noisy, x_trunk)
                else:  # ortho_layer_ica
                    output, _ = model(x_branch_noisy, x_trunk)
                
                output = output.squeeze(-1)
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate and calculate metrics
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mse = np.mean((predictions - targets)**2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Relative L2 error
    diff_norm = np.linalg.norm(predictions - targets, axis=1)
    true_norm = np.linalg.norm(targets, axis=1)
    rel_l2_errors = diff_norm / (true_norm + 1e-10)
    rel_l2 = np.mean(rel_l2_errors)
    rel_l2_std = np.std(rel_l2_errors)
    
    return {
        'mse': mse,
        'mae': mae,
        'rel_l2': rel_l2,
        'rel_l2_std': rel_l2_std
    }

def calculate_input_statistics(data_loader, is_fno=False):
    """Calculate input data statistics for noise scaling."""
    
    all_inputs = []
    
    for batch_data in data_loader:
        if is_fno:
            x, _ = batch_data
            # Extract permeability field (first channel)
            all_inputs.append(x[:, :, :, 0].reshape(x.shape[0], -1))
        else:
            x_branch, _ = batch_data[0]
            all_inputs.append(x_branch)
    
    full_data = torch.cat(all_inputs, dim=0)
    data_std = full_data.std().item()
    data_mean = full_data.mean().item()
    
    return data_std, data_mean

# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("         DARCY NOISE ROBUSTNESS EVALUATION")
    print("="*70)
    
    # Load data
    data_dict = load_data(config)
    
    # Get all model types
    model_types = config['models']['types_don'] + config['models']['types_fno']
    seeds = config['training']['seeds']
    
    # Noise testing parameters
    noise_types = config['noise_testing']['types']
    noise_levels = config['noise_testing']['levels']
    noise_fractions = config['noise_testing']['fractions']
    
    # Results storage
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"EVALUATING MODELS - SEED {seed}")
        print(f"{'='*60}")
        
        seed_results = []
        
        for model_type in model_types:
            print(f"\n📊 Evaluating {model_type}...")
            
            # Load model
            model = create_model(model_type, config, data_dict['train_data_res'])
            model_path = Path(config['paths']['save_models']) / f"best_model_{model_type}_seed{seed}.pth"
            
            if not model_path.exists():
                print(f"  ⚠️  Model file not found: {model_path}")
                continue
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            # Get appropriate data loader
            fno_types = ['fourier', 'fourier_layer_ortho', 'wavelet']
            is_fno = model_type in fno_types
            
            if is_fno:
                test_loader = data_dict['test_loader_fno']
            else:
                test_loader = data_dict['test_loader_don']
            
            # Calculate input data statistics
            print("  Calculating input statistics...")
            input_data_std, _ = calculate_input_statistics(test_loader, is_fno)
            print(f"  Input StdDev: {input_data_std:.4f}")
            
            # Test different noise configurations
            for noise_type in noise_types:
                for noise_level in noise_levels:
                    for noise_fraction in noise_fractions:
                        
                        # Skip redundant configurations
                        if noise_level == 0.0 and noise_fraction != 1.0:
                            # Copy baseline result
                            base_result = [r for r in seed_results 
                                         if r['model_type'] == model_type 
                                         and r['noise_level'] == 0.0 
                                         and r['noise_type'] == noise_type]
                            if base_result:
                                new_result = base_result[0].copy()
                                new_result['noise_fraction'] = noise_fraction
                                seed_results.append(new_result)
                            continue
                        
                        if noise_type != 'partial_random' and noise_fraction != 1.0:
                            continue
                        
                        # Evaluate
                        metrics = evaluate_with_noise(
                            model, test_loader, 
                            noise_type, noise_level, noise_fraction,
                            config, model_type, input_data_std
                        )
                        
                        # Store results
                        result = {
                            'seed': seed,
                            'model_type': model_type,
                            'noise_type': noise_type,
                            'noise_level': noise_level,
                            'noise_fraction': noise_fraction,
                            **metrics
                        }
                        
                        seed_results.append(result)
                        
                        # Print progress
                        print(f"    {noise_type:20s} | σ={noise_level:.1f} | f={noise_fraction:.0%} | L2={metrics['rel_l2']:.4e}")
        
        # Save results for this seed
        results_df = pd.DataFrame(seed_results)
        save_path = Path(config['paths']['save_figures'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        results_file = save_path / f'analysis_results_seed_{seed}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ Results saved to: {results_file}")
        
        all_results.extend(seed_results)
    
    # Save aggregated results
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_file = save_path / 'all_results.csv'
        all_results_df.to_csv(all_results_file, index=False)
        
        # Print summary statistics
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        summary = all_results_df.groupby(['model_type', 'noise_type', 'noise_level'])['rel_l2'].agg(['mean', 'std'])
        print("\nMean Relative L2 Error (across all seeds):")
        print(summary.to_string())
    
    print("\n🎉 Evaluation complete!")

if __name__ == '__main__':
    main()