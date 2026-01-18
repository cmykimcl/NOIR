#!/usr/bin/env python
"""
Evaluate noise robustness of trained Burgers equation models.
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

from noise_generators import NoiseGeneratorBurger
from train import load_data, create_model
from losses import ICALoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_with_noise(model, test_loader, noise_type, noise_level, noise_fraction,
                       config, model_type, input_data_std):
    """Evaluate model performance with specific noise configuration."""
    
    model.eval()
    
    # Check if model is FNO-type or DeepONet-type
    fno_types = ['fourier', 'fourier_ortho', 'wavelet']
    is_fno = model_type in fno_types
    
    s = config['data']['resolution']
    orthogonal_loss_fn = ICALoss()
    
    all_predictions = []
    all_targets = []
    all_ortho_scores = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if is_fno:
                x, y = batch_data
                x_func = x[:, :, 0]  # Initial condition
                x_coord = x[:, :, 1:].to(device)
                y = y.to(device)
                
                # Apply noise to initial condition
                batch_size = x_func.shape[0]
                generator_shape = (batch_size, s)
                noise_gen = NoiseGeneratorBurger(x_shape=generator_shape, device=device)
                
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
                else:  # fourier_ortho
                    output, x_ = model(x_noisy_full)
                    B, W, C_out, p_ica = x_.shape
                    x_ = x_.reshape(B * W, C_out, p_ica)
                    ortho_loss = orthogonal_loss_fn(x_)
                    all_ortho_scores.append(ortho_loss.cpu().numpy())
                
            else:  # DeepONet models
                x_branch, x_trunk = batch_data[0]
                y = batch_data[1]
                x_trunk = x_trunk.to(device)
                y = y.to(device)
                
                # Apply noise to branch input
                batch_size = x_branch.shape[0]
                x_branch_flat = x_branch.squeeze(-1)
                generator_shape = (batch_size, s)
                noise_gen = NoiseGeneratorBurger(x_shape=generator_shape, device=device)
                
                if noise_level > 0:
                    noise_std = noise_level * input_data_std
                    x_branch_noisy, _ = noise_gen.generate(
                        x=x_branch_flat.to(device),
                        noise_type=noise_type,
                        noise_std=noise_std,
                        k_fraction=noise_fraction
                    )
                    x_branch_noisy = x_branch_noisy.unsqueeze(-1)
                else:
                    x_branch_noisy = x_branch.to(device)
                
                # Get prediction
                if model_type in ['default', 'nomad']:
                    output = model(x_branch_noisy, x_trunk)
                else:  # ortho_ica
                    output, x_ = model(x_branch_noisy, x_trunk)
                    B, P, C, d = x_.shape
                    x_ = x_.reshape(B * P, C, d)
                    ortho_loss = orthogonal_loss_fn(x_)
                    all_ortho_scores.append(ortho_loss.cpu().numpy())
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate and calculate metrics
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Reshape if needed
    if predictions.ndim == 3:
        predictions = predictions.squeeze(-1)
    if targets.ndim == 3:
        targets = targets.squeeze(-1)
    
    # Calculate metrics
    mse = np.mean((predictions - targets)**2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Relative L2 error
    diff_norm = np.linalg.norm(predictions - targets, axis=1)
    true_norm = np.linalg.norm(targets, axis=1)
    rel_l2_errors = diff_norm / (true_norm + 1e-10)
    rel_l2 = np.mean(rel_l2_errors)
    rel_l2_std = np.std(rel_l2_errors)
    
    # Average orthogonality score
    avg_ortho_score = np.mean(all_ortho_scores) if all_ortho_scores else np.nan
    std_ortho_score = np.std(all_ortho_scores) if all_ortho_scores else np.nan
    
    return {
        'mse': mse,
        'mae': mae,
        'rel_l2': rel_l2,
        'rel_l2_std': rel_l2_std,
        'avg_ortho_score': avg_ortho_score,
        'std_ortho_score': std_ortho_score
    }

def calculate_input_statistics(data_dict):
    """Calculate input data statistics for noise scaling."""
    train_input = data_dict['train_input']
    data_std = train_input.std().item()
    data_mean = train_input.mean().item()
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
    print("         BURGERS EQUATION NOISE ROBUSTNESS EVALUATION")
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
    
    # Calculate input data statistics
    print("\n📊 Calculating input statistics...")
    input_data_std, input_data_mean = calculate_input_statistics(data_dict)
    print(f"  Input StdDev: {input_data_std:.4f}")
    print(f"  Input Mean: {input_data_mean:.4f}")
    
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
            model = create_model(model_type, config)
            model_path = Path(config['paths']['save_models']) / f"best_model_{model_type}_seed{seed}.pth"
            
            if not model_path.exists():
                print(f"  ⚠️  Model file not found: {model_path}")
                continue
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            # Get appropriate data loader
            fno_types = ['fourier', 'fourier_ortho', 'wavelet']
            is_fno = model_type in fno_types
            
            if is_fno:
                test_loader = data_dict['test_loader_fno']
            else:
                test_loader = data_dict['test_loader_don']
            
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
                        ortho_str = f" | Ortho={metrics['avg_ortho_score']:.4e}" if not np.isnan(metrics['avg_ortho_score']) else ""
                        print(f"    {noise_type:20s} | σ={noise_level:.1f} | f={noise_fraction:.0%} | L2={metrics['rel_l2']:.4e}{ortho_str}")
        
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
        
        # Calculate aggregated statistics
        agg_results = all_results_df.groupby(['model_type', 'noise_type', 'noise_level']).agg({
            'rel_l2': ['mean', 'std'],
            'avg_ortho_score': ['mean', 'std']
        }).reset_index()
        
        agg_results.columns = ['_'.join(col).strip('_') for col in agg_results.columns.values]
        agg_results_file = save_path / 'aggregated_results_all_seeds.csv'
        agg_results.to_csv(agg_results_file, index=False)
        
        # Print summary statistics
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        summary = all_results_df.groupby(['model_type', 'noise_type', 'noise_level'])['rel_l2'].agg(['mean', 'std'])
        print("\nMean Relative L2 Error (across all seeds):")
        print(summary.to_string())
        
        # Print orthogonality scores for relevant models
        ortho_models = ['ortho_ica', 'fourier_ortho']
        ortho_df = all_results_df[all_results_df['model_type'].isin(ortho_models)]
        if not ortho_df.empty:
            print("\n" + "="*50)
            print("ORTHOGONALITY SCORES")
            print("="*50)
            ortho_summary = ortho_df.groupby(['model_type', 'noise_level'])['avg_ortho_score'].agg(['mean', 'std'])
            print(ortho_summary.to_string())
    
    print("\n🎉 Evaluation complete!")

if __name__ == '__main__':
    main()