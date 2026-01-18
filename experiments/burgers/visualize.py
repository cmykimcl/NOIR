#!/usr/bin/env python
"""
Generate publication-quality figures from Burgers equation experiment results.
Usage: python visualize.py [--config config.yaml]
"""

import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
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
        'ortho_ica': 'DeepONet+NOIR',
        'fourier_ortho': 'FNO+NOIR',
    }
    combined_df['model_display'] = combined_df['model_type'].map(model_name_map).fillna(combined_df['model_type'])
    
    return combined_df

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
    
    plt.suptitle('Training History - Burgers Equation', fontsize=14, fontweight='bold')
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
    
    # Create comparison table for partial noise
    df_partial = df[df['noise_type'] == 'partial_random']
    if not df_partial.empty:
        partial_pivot = df_partial.pivot_table(
            values='rel_l2',
            index='model_display',
            columns='noise_fraction',
            aggfunc='mean'
        )
        
        partial_latex = save_path / 'table_partial_noise_fractions.tex'
        with open(partial_latex, 'w') as f:
            f.write(partial_pivot.to_latex(float_format="%.4f"))
        print(f"✅ Saved: {partial_latex}")

def plot_model_comparison_table(df, config, save_path):
    """Create a comprehensive comparison table figure."""
    
    # Calculate statistics for each model
    stats_data = []
    
    for model in df['model_display'].unique():
        model_df = df[df['model_display'] == model]
        
        # Baseline performance (no noise)
        baseline = model_df[model_df['noise_level'] == 0.0]['rel_l2'].mean()
        
        # Performance at different noise levels
        perf_01 = model_df[model_df['noise_level'] == 0.1]['rel_l2'].mean()
        perf_05 = model_df[model_df['noise_level'] == 0.5]['rel_l2'].mean()
        perf_10 = model_df[model_df['noise_level'] == 1.0]['rel_l2'].mean()
        
        # Degradation ratios
        deg_01 = perf_01 / baseline if baseline > 0 else np.nan
        deg_05 = perf_05 / baseline if baseline > 0 else np.nan
        deg_10 = perf_10 / baseline if baseline > 0 else np.nan
        
        stats_data.append({
            'Model': model,
            'Baseline': baseline,
            'σ=0.1': perf_01,
            'σ=0.5': perf_05,
            'σ=1.0': perf_10,
            'Deg. 0.1': deg_01,
            'Deg. 0.5': deg_05,
            'Deg. 1.0': deg_10
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Format values for display
    cell_text = []
    for _, row in stats_df.iterrows():
        formatted_row = [
            row['Model'],
            f"{row['Baseline']:.4f}",
            f"{row['σ=0.1']:.4f}",
            f"{row['σ=0.5']:.4f}",
            f"{row['σ=1.0']:.4f}",
            f"{row['Deg. 0.1']:.2f}×",
            f"{row['Deg. 0.5']:.2f}×",
            f"{row['Deg. 1.0']:.2f}×"
        ]
        cell_text.append(formatted_row)
    
    # Create table
    table = ax.table(cellText=cell_text,
                    colLabels=stats_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color cells based on performance
    for i in range(1, len(cell_text) + 1):
        # Highlight best baseline
        if i <= len(cell_text):
            for j in range(1, 5):
                val = float(cell_text[i-1][j].strip('×'))
                if j == 1:  # Baseline column
                    if val == stats_df['Baseline'].min():
                        table[(i, j)].set_facecolor('#E8F5E9')
                else:  # Noise columns
                    col_name = stats_df.columns[j]
                    if val == stats_df[col_name].min():
                        table[(i, j)].set_facecolor('#E8F5E9')
    
    plt.title('Model Performance Comparison - Burgers Equation', 
             fontsize=14, fontweight='bold', pad=20)
    
    save_file = save_path / 'model_comparison_table.png'
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
    print("         BURGERS EQUATION VISUALIZATION")
    print("="*70)
    
    # Create output directory
    save_path = Path(config['paths']['save_figures'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load results
        print("\n📊 Loading results...")
        df = load_results(config)
        print(f"  Loaded {len(df)} results")
        
        print("\n📈 Generating plots...")
        
        print("  Creating error sensitivity plots...")
        plot_error_sensitivity(df, config, save_path)
        
        print("  Creating training history plots...")
        plot_training_history(config, save_path)
        
        print("  Creating model comparison table...")
        plot_model_comparison_table(df, config, save_path)
        
        print("  Creating summary tables...")
        create_summary_tables(df, config, save_path)
        
        print("\n✅ All visualizations complete!")
        print(f"Figures saved to: {save_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run evaluate.py first to generate results.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()