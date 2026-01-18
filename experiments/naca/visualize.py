#!/usr/bin/env python
"""
Generate publication-quality figures from NACA experiment results.
Usage: python visualize.py [--config config.yaml]
"""

import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(config):
    """Load evaluation results from CSV."""
    figures_path = Path(config['paths']['save_figures'])
    
    all_results = []
    for seed in config['training']['seeds']:
        results_file = figures_path / f'analysis_results_seed_{seed}.csv'
        if results_file.exists():
            df = pd.read_csv(results_file)
            all_results.append(df)
    
    if not all_results:
        raise FileNotFoundError("No results files found. Run evaluate.py first!")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Map model names for better display
    model_name_map = {
        'default': 'DeepONet',
        'fourier': 'FNO',
        'nomad': 'NOMAD',
        'ortho_layer_ica': 'DeepONet+NOIR',
        'fourier_layer_ortho': 'FNO+NOIR',
    }
    combined_df['model_display'] = combined_df['model_type'].map(model_name_map).fillna(combined_df['model_type'])
    
    return combined_df

def plot_noise_robustness_comparison(df, config, save_path):
    """Create main comparison plot for all models and noise types."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    noise_types = df['noise_type'].unique()
    
    for idx, noise_type in enumerate(noise_types):
        if idx >= 6:
            break
            
        ax = axes[idx]
        df_noise = df[df['noise_type'] == noise_type]
        
        # Group by model and noise level, calculate statistics
        grouped = df_noise.groupby(['model_display', 'noise_level']).agg({
            'rel_l2': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['model', 'noise_level', 'mean', 'std']
        
        # Plot each model
        for model in grouped['model'].unique():
            model_data = grouped[grouped['model'] == model]
            ax.errorbar(model_data['noise_level'], 
                       model_data['mean'],
                       yerr=model_data['std'],
                       marker='o', 
                       label=model,
                       linewidth=2,
                       markersize=8,
                       capsize=5)
        
        ax.set_xlabel('Noise Level (σ)', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontsize=12)
        ax.set_title(noise_type.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    # Remove empty subplot if exists
    if len(noise_types) < 6:
        fig.delaxes(axes[-1])
    
    plt.suptitle('Noise Robustness Comparison - NACA Airfoil', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_file = save_path / 'noise_robustness_comparison.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.savefig(save_file.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✅ Saved: {save_file}")
    
    plt.close()

def plot_error_sensitivity(df, config, save_path):
    """Plot error sensitivity (relative error increase)."""
    
    unique_noise = df['noise_type'].unique()
    
    for noise_type in unique_noise:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        df_noise = df[df['noise_type'] == noise_type]
        
        # Calculate error sensitivity
        for model in df_noise['model_display'].unique():
            model_data = df_noise[df_noise['model_display'] == model].copy()
            
            # Group by noise level and calculate mean
            grouped = model_data.groupby('noise_level')['rel_l2'].agg(['mean', 'std']).reset_index()
            # Calculate sensitivity relative to baseline (no noise)
            baseline = grouped[grouped['noise_level'] == 0.0]['mean'].values[0]
            grouped['sensitivity'] = grouped['mean'] / baseline
            grouped['sensitivity_std'] = grouped['std'] / baseline
            
            # Plot with error bars
            ax.errorbar(grouped['noise_level'], 
                       grouped['sensitivity'],
                       yerr=grouped['sensitivity_std'],
                       marker='o',
                       label=model,
                       linewidth=2.5,
                       markersize=10,
                       capsize=6,
                       capthick=2)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Noise Level (σ/σ_data)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Relative Error Increase', fontsize=14, fontweight='bold')
        ax.set_title(f'Error Sensitivity - {noise_type.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='upper left', fontsize=12, frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_file = save_path / f'error_sensitivity_{noise_type}.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.savefig(save_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"✅ Saved: {save_file}")
        
        plt.close()

def plot_sample_predictions(config, save_path):
    """Plot sample predictions with and without noise."""
    
    import torch
    import sys
    sys.path.append('../../src')
    
    from train import load_data, create_model
    
    # Load data
    data_dict = load_data(config)
    
    # Load one model for visualization
    model_type = 'default'
    seed = config['training']['seeds'][0]
    
    model = create_model(model_type, config, data_dict['branch_input_size'])
    model_path = Path(config['paths']['save_models']) / f"best_model_{model_type}_seed{seed}.pth"
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Get a sample from test set
        test_loader = data_dict['test_loader_don']
        sample_batch = next(iter(test_loader))
        
        with torch.no_grad():
            x_branch, x_trunk = sample_batch[0]
            y_true = sample_batch[1]
            
            # Get prediction
            y_pred = model(x_branch[:1], x_trunk[:1])
            
            # Reshape for visualization
            s1, s2 = data_dict['s1'], data_dict['s2']
            y_true = y_true[0].reshape(s1, s2).numpy()
            y_pred = y_pred[0].reshape(s1, s2).numpy()
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Ground truth
            im1 = axes[0].imshow(y_true, cmap='jet', aspect='auto')
            axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Prediction
            im2 = axes[1].imshow(y_pred, cmap='jet', aspect='auto')
            axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Error
            error = np.abs(y_true - y_pred)
            im3 = axes[2].imshow(error, cmap='Reds', aspect='auto')
            axes[2].set_title('Absolute Error', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('y')
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            
            rel_error = np.linalg.norm(error) / np.linalg.norm(y_true)
            plt.suptitle(f'{model_type} - Sample Prediction (Rel. L2 Error: {rel_error:.3%})', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            save_file = save_path / f'sample_prediction_{model_type}.png'
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_file}")
            
            plt.close()

def create_summary_table(df, config, save_path):
    """Create summary table of results."""
    
    # Calculate mean performance across seeds
    summary = df.groupby(['model_display', 'noise_type', 'noise_level']).agg({
        'rel_l2': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # Save as CSV
    summary_file = save_path / 'summary_table.csv'
    summary.to_csv(summary_file)
    print(f"✅ Saved: {summary_file}")
    
    # Create LaTeX table for paper
    for noise_type in df['noise_type'].unique():
        df_noise = df[df['noise_type'] == noise_type]
        
        pivot_table = df_noise.pivot_table(
            values='rel_l2',
            index='model_display',
            columns='noise_level',
            aggfunc='mean'
        )
        
        latex_file = save_path / f'table_{noise_type}.tex'
        with open(latex_file, 'w') as f:
            f.write(pivot_table.to_latex(float_format="%.4f"))
        print(f"✅ Saved: {latex_file}")

def plot_training_history(config, save_path):
    """Plot training history for all models."""
    
    log_path = Path(config['paths']['save_logs'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    all_models = config['models']['types_don'] + config['models']['types_fno']
    
    for idx, model_type in enumerate(all_models):
        if idx >= 6:
            break
            
        ax = axes[idx]
        
        # Load training histories for all seeds
        for seed in config['training']['seeds']:
            train_file = log_path / f"train_loss_{model_type}_seed{seed}.npy"
            val_file = log_path / f"val_loss_{model_type}_seed{seed}.npy"
            
            if train_file.exists() and val_file.exists():
                train_loss = np.load(train_file)
                val_loss = np.load(val_file)
                
                epochs = np.arange(1, len(train_loss) + 1)
                
                ax.plot(epochs, train_loss, alpha=0.3, color='blue')
                ax.plot(epochs, val_loss, alpha=0.3, color='red')
        
        # Add mean lines
        all_train = []
        all_val = []
        for seed in config['training']['seeds']:
            train_file = log_path / f"train_loss_{model_type}_seed{seed}.npy"
            val_file = log_path / f"val_loss_{model_type}_seed{seed}.npy"
            if train_file.exists() and val_file.exists():
                all_train.append(np.load(train_file))
                all_val.append(np.load(val_file))
        
        if all_train:
            # Truncate to minimum length
            min_len = min(len(t) for t in all_train)
            all_train = [t[:min_len] for t in all_train]
            all_val = [v[:min_len] for v in all_val]
            
            mean_train = np.mean(all_train, axis=0)
            mean_val = np.mean(all_val, axis=0)
            
            epochs = np.arange(1, len(mean_train) + 1)
            ax.plot(epochs, mean_train, 'b-', linewidth=2, label='Train (mean)')
            ax.plot(epochs, mean_val, 'r-', linewidth=2, label='Val (mean)')
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(model_type, fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle('Training History - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_file = save_path / 'training_history.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("         NACA VISUALIZATION")
    print("="*70)
    
    # Create output directory
    save_path = Path(config['paths']['save_figures'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load results
        print("\n📊 Loading results...")
        df = load_results(config)
        print(f"  Loaded {len(df)} results")
        
        # Generate plots
        print("\n📈 Generating plots...")
        
        print("  Creating noise robustness comparison...")
        plot_noise_robustness_comparison(df, config, save_path)
        
        print("  Creating error sensitivity plots...")
        plot_error_sensitivity(df, config, save_path)
        
        print("  Creating sample predictions...")
        plot_sample_predictions(config, save_path)
        
        print("  Creating summary tables...")
        create_summary_table(df, config, save_path)
        
        print("  Creating training history plots...")
        plot_training_history(config, save_path)
        
        print("\n✅ All visualizations complete!")
        print(f"Figures saved to: {save_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run evaluate.py first to generate results.")
    
if __name__ == '__main__':
    main()