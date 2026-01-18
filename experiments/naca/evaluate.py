#!/usr/bin/env python
"""
Evaluate noise robustness of trained NACA models.
Usage: python evaluate.py [--config config.yaml]
"""
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append('../../src')

from noise_generators import NoiseGeneratorNACA

def evaluate_with_noise(model, test_loader, noise_type, noise_level, config):
    """Evaluate model performance with specific noise."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    noise_gen = NoiseGeneratorNACA(...)
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Apply noise
            noisy_input = noise_gen.generate(batch, noise_type, noise_level)
            
            # Get predictions
            output = model(noisy_input)
            
            predictions.append(output.cpu().numpy())
            targets.append(batch[1].cpu().numpy())
    
    # Calculate metrics
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    rel_l2_error = np.linalg.norm(predictions - targets) / np.linalg.norm(targets)
    
    return {'rel_l2': rel_l2_error}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    _, test_loader = load_data(config)
    
    results = []
    
    # Evaluate each model
    for seed in config['training']['seeds']:
        for model_type in config['models']['types_don'] + config['models']['types_fno']:
            # Load model
            model_path = Path(config['paths']['save_models']) / f'{model_type}_seed{seed}_best.pth'
            model = create_model(model_type, config)
            model.load_state_dict(torch.load(model_path))
            
            # Test with different noise conditions
            for noise_type in config['noise_testing']['types']:
                for noise_level in config['noise_testing']['levels']:
                    metrics = evaluate_with_noise(model, test_loader, 
                                                 noise_type, noise_level, config)
                    
                    results.append({
                        'seed': seed,
                        'model': model_type,
                        'noise_type': noise_type,
                        'noise_level': noise_level,
                        **metrics
                    })
                    
                    print(f"{model_type} | {noise_type} | σ={noise_level:.2f} | L2={metrics['rel_l2']:.4e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    save_path = Path(config['paths']['save_figures']) / 'evaluation_results.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=False)
    
    print(f"\n✅ Evaluation complete! Results saved to {save_path}")

if __name__ == '__main__':
    main()