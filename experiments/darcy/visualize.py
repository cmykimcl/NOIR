#!/usr/bin/env python
"""
Generate publication-quality figures from Darcy experiment results.
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
import torch

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(config):
    """Load evaluation results from CSV files."""
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
        'wavelet': 'WNO',
        'ortho_layer_ica': 'DeepONet+NOIR',
        'fourier_layer_ortho': 'FNO+NOIR',
    }
    combined_df['model_display'] = combined_df['model_type'].map(model_name_map).fillna(combined_df['model_type'])
    
    return combined_df

def plot_noise_robustness_grid(df, config, save_path):
    """Create grid plot showing all noise types and models."""
    
    # Create figure with subplots for each noise type
    noise_types = df['noise_type'].unique()
    n_noise = len(noise_types)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, noise_type in enumerate(noise_types):
        if idx >= 6:
            break
        
        ax = axes[idx]
        df_noise = df[df['noise_type'] == noise_type]
        
        # Group by model and noise level
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
                       linewidth=2.5,
                       markersize=8,
                       capsize=5,
                       capthick=2,
                       alpha=0.8)
        
        ax.set_xlabel('Noise Level (σ)', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontsize=12)
        ax.set_title(noise_type.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # Remove empty subplot if needed
    if len(noise_types) < 6:
        fig.delaxes(axes[-1])
    
    plt.suptitle('Noise Robustness Analysis - Darcy Flow', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_file = save_path / 'noise_robustness_grid.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.savefig(save_file.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✅ Saved: {save_file}")
    
    plt.close()

def plot_error_sensitivity(df, config, save_path):
    """Plot relative error increase for each noise type."""
    
    unique_noise = df['noise_type'].unique()
    
    for noise_type in unique_noise:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        df_noise = df[df['noise_type'] == noise_type]
        
        # Calculate error sensitivity (relative to no-noise baseline)
        for model in df_noise['model_display'].unique():
            model_data = df_noise[df_noise['model_display'] == model].copy()
            
            # Group by noise level
            grouped = model_data.groupby('noise_level')['rel_l2'].agg(['mean', 'std']).reset_index()
            
            # Get baseline (no noise)
            baseline = grouped[grouped['noise_level'] == 0.0]['mean'].values[0]
            
            # Calculate sensitivity
            grouped['sensitivity'] = grouped['mean'] / baseline
            grouped['sensitivity_std'] = grouped['std'] / baseline
            
            # Plot
            ax.errorbar(grouped['noise_level'],
                       grouped['sensitivity'],
                       yerr=grouped['sensitivity_std'],
                       marker='o',
                       label=model,
                       linewidth=2.5,
                       markersize=10,
                       capsize=6,
                       capthick=2,
                       alpha=0.8)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
        ax.set_xlabel('Noise Level (σ/σ_data)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Relative Error Increase', fontsize=14, fontweight='bold')
        ax.set_title(f'Error Sensitivity - {noise_type.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_file = save_path / f'error_sensitivity_{noise_type}.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.savefig(save_file.with_suffix('.pdf'), bbox_inches='tight')
        print(f"✅ Saved: {save_file}")
        
        plt.close()

def plot_sample_predictions(config, save_path):
    """Visualize sample predictions from different models."""
    
    import sys
    sys.path.append('.')
    from utils_darcy import load_data, create_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_dict = load_data(config)
    s = data_dict['s']
    
    # Models to visualize
    models_to_show = ['default', 'ortho_layer_ica']
    seed = config['training']['seeds'][0]
    
    # Create figure
    n_models = len(models_to_show)
    fig, axes = plt.subplots(n_models, 3, figsize=(12, 4*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, model_type in enumerate(models_to_show):
        # Load model
        model = create_model(model_type, config, data_dict['train_data_res'])
        model_path = Path(config['paths']['save_models']) / f"best_model_{model_type}_seed{seed}.pth"
        
        if not model_path.exists():
            print(f"  ⚠️  Model not found: {model_path}")
            continue
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Get a test sample
        test_loader = data_dict['test_loader_don']
        sample_batch = next(iter(test_loader))
        
        with torch.no_grad():
            x_branch, x_trunk = sample_batch[0]
            y_true = sample_batch[1]
            
            x_branch = x_branch[:1].to(device)
            x_trunk = x_trunk[:1].to(device)
            y_true = y_true[:1]
            
            # Get prediction
            if model_type in ['default', 'nomad']:
                y_pred = model(x_branch, x_trunk)
            else:
                y_pred, _ = model(x_branch, x_trunk)
            
            # Reshape for visualization
            x_input = x_branch[0].cpu().reshape(s, s).numpy()
            y_true = y_true[0].reshape(s, s).numpy()
            y_pred = y_pred[0].cpu().reshape(s, s).numpy()
            
            # Plot input (permeability field)
            im1 = axes[idx, 0].imshow(x_input, cmap='viridis', aspect='auto')
            axes[idx, 0].set_title('Input (Permeability)', fontsize=12, fontweight='bold')
            axes[idx, 0].set_ylabel(model_type.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            plt.colorbar(im1, ax=axes[idx, 0], fraction=0.046, pad=0.04)
            
            # Plot ground truth
            im2 = axes[idx, 1].imshow(y_true, cmap='jet', aspect='auto')
            axes[idx, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            plt.colorbar(im2, ax=axes[idx, 1], fraction=0.046, pad=0.04)
            
            # Plot prediction
            im3 = axes[idx, 2].imshow(y_pred, cmap='jet', aspect='auto')
            axes[idx, 2].set_title('Prediction', fontsize=12, fontweight='bold')
            plt.colorbar(im3, ax=axes[idx, 2], fraction=0.046, pad=0.04)
            
            # Calculate error
            rel_error = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)
            axes[idx, 2].text(0.95, 0.05, f'Rel. L2: {rel_error:.3%}',
                             transform=axes[idx, 2].transAxes,
                             ha='right', va='bottom',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Sample Predictions - Darcy Flow', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_file = save_path / 'sample_predictions.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_file}")
    
    plt.close()

def plot_training_history(config, save_path):
    """Plot training history for all models."""
    
    log_path = Path(config['paths']['save_logs'])
    
    # Get all model types
    model_types = config['models']['types_don'] + config['models']['types_fno']
    n_models = len(model_types)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, model_type in enumerate(model_types):
        if idx >= 6:
            break
        
        ax = axes[idx]
        
        # Load and plot training history for each seed
        for seed in config['training']['seeds']:
            train_file = log_path / f"train_loss_{model_type}_seed{seed}.npy"
            val_file = log_path / f"val_loss_{model_type}_seed{seed}.npy"
            
            if train_file.exists() and val_file.exists():
                train_loss = np.load(train_file)
                val_loss = np.load(val_file)
                
                epochs = np.arange(1, len(train_loss) + 1)
                
                # Plot with transparency for individual seeds
                ax.plot(epochs, train_loss, 'b-', alpha=0.2, linewidth=0.5)
                ax.plot(epochs, val_loss, 'r-', alpha=0.2, linewidth=0.5)
        
        # Calculate and plot mean
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
        ax.set_title(model_type.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # Remove empty subplot if needed
    if len(model_types) < 6:
        fig.delaxes(axes[-1])
    
    plt.suptitle('Training History - Darcy Flow', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_file = save_path / 'training_history.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_file}")
    
    plt.close()

def create_summary_tables(df, config, save_path):
    """Create summary tables for the paper."""
    
    # Summary statistics by model and noise type
    summary = df.groupby(['model_display', 'noise_type', 'noise_level']).agg({
        'rel_l2': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # Save as CSV
    summary_file = save_path / 'summary_statistics.csv'
    summary.to_csv(summary_file)
    print(f"✅ Saved: {summary_file}")
    
    # Create LaTeX tables for each noise type
    for noise_type in df['noise_type'].unique():
        df_noise = df[df['noise_type'] == noise_type]
        
        # Pivot table for this noise type
        pivot = df_noise.pivot_table(
            values='rel_l2',
            index='model_display',
            columns='noise_level',
            aggfunc='mean'
        )
        
        # Save as LaTeX
        latex_file = save_path / f'table_{noise_type}.tex'
        with open(latex_file, 'w') as f:
            f.write(pivot.to_latex(float_format="%.4f"))
        print(f"✅ Saved: {latex_file}")
    
    # Create a master table at noise level 0.5
    df_05 = df[df['noise_level'] == 0.5]
    master_pivot = df_05.pivot_table(
        values='rel_l2',
        index='model_display',
        columns='noise_type',
        aggfunc='mean'
    )
    
    master_latex = save_path / 'master_table_noise_05.tex'
    with open(master_latex, 'w') as f:
        f.write(master_pivot.to_latex(float_format="%.4f"))
    print(f"✅ Saved: {master_latex}")

def plot_comparison_figure(df, config, save_path):
    """Create main comparison figure for the paper."""
    
    # Select two representative noise types
    noise_types_show = ['partial_random', 'harmonic_spatial']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, noise_type in enumerate(noise_types_show):
        ax = axes[idx]
        df_noise = df[df['noise_type'] == noise_type]
        
        # Calculate sensitivity
        for model in df_noise['model_display'].unique():
            model_data = df_noise[df_noise['model_display'] == model].copy()
            
            grouped = model_data.groupby('noise_level')['rel_l2'].agg(['mean', 'std']).reset_index()
            
            baseline = grouped[grouped['noise_level'] == 0.0]['mean'].values[0]
            grouped['sensitivity'] = grouped['mean'] / baseline
            grouped['sensitivity_std'] = grouped['std'] / baseline
            
            # Define colors for consistency
            color_map = {
                'DeepONet': '#2E86AB',
                'FNO': '#F18F01',
                'NOMAD': '#6A994E',
                'DeepONet+NOIR': '#A23B72',
                'FNO+NOIR': '#C73E1D'
            }
            
            ax.errorbar(grouped['noise_level'],
                       grouped['sensitivity'],
                       yerr=grouped['sensitivity_std'],
                       marker='o',
                       label=model,
                       color=color_map.get(model, 'gray'),
                       linewidth=2.5,
                       markersize=10,
                       capsize=6,
                       capthick=2,
                       alpha=0.8)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Noise Level (σ/σ_data)', fontsize=13, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Relative Error Increase', fontsize=13, fontweight='bold')
        ax.set_title(noise_type.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Darcy Flow - Noise Robustness Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    save_file = save_path / 'main_comparison.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.savefig(save_file.with_suffix('.pdf'), bbox_inches='tight')
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
    print("         DARCY VISUALIZATION")
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
        
        print("  Creating noise robustness grid...")
        plot_noise_robustness_grid(df, config, save_path)
        
        print("  Creating error sensitivity plots...")
        plot_error_sensitivity(df, config, save_path)
        
        print("  Creating sample predictions...")
        plot_sample_predictions(config, save_path)
        
        print("  Creating training history plots...")
        plot_training_history(config, save_path)
        
        print("  Creating summary tables...")
        create_summary_tables(df, config, save_path)
        
        print("  Creating main comparison figure...")
        plot_comparison_figure(df, config, save_path)
        
        print("\n✅ All visualizations complete!")
        print(f"Figures saved to: {save_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run evaluate.py first to generate results.")
    
if __name__ == '__main__':
    main()