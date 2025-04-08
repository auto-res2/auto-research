"""
Main script for running the ADNLCC (Ambient Diffusion with Non-Linear Characteristic Correction) experiment.

This script orchestrates the entire experiment workflow:
1. Data preprocessing
2. Model training (Base method and ADNLCC method)
3. Model evaluation and comparison
4. Visualization of results

All figures and plots are saved in high-quality PDF format suitable for academic papers.
The code is optimized to run on NVIDIA Tesla T4 GPUs with 16GB VRAM.
"""

import torch
import torch.nn as nn
import os
import argparse
import json
import time
from datetime import datetime

from preprocess import get_dataloaders
from train import SimpleDiffusionModel, train_model
from evaluate import evaluate_model

def create_config(args):
    """Create configuration dictionary from command line arguments."""
    config = {
        'in_channels': 3,
        'base_channels': 64,
        
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'lr_step_size': 30,
        'lr_gamma': 0.5,
        
        'noise_level': args.noise_level,
        
        'sampling_steps': 50,
        
        'device': args.device,
        
        'dataset': 'CIFAR10',
        
        'test_mode': args.test_mode
    }
    return config

def save_config(config, filename='config.json'):
    """Save configuration to a JSON file."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return config_path

def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU available, using CPU")

def run_experiment(config):
    """Run the ADNLCC experiment with the given configuration."""
    print("\n" + "="*80)
    print(f"Starting ADNLCC experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\nExperiment Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nHardware Information:")
    print_gpu_info()
    
    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")
    
    print("\nPreparing datasets...")
    train_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        dataset_name=config['dataset'],
        add_noise=False,  # We'll add noise during training
        noise_level=config['noise_level']
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    print("\nInitializing models...")
    model_base = SimpleDiffusionModel(
        in_channels=config['in_channels'],
        base_channels=config['base_channels']
    )
    model_adnlcc = SimpleDiffusionModel(
        in_channels=config['in_channels'],
        base_channels=config['base_channels']
    )
    
    num_epochs = 1 if config['test_mode'] else config['num_epochs']
    
    print("\n" + "-"*80)
    print("Training Base Model (without ADNLCC)...")
    print("-"*80)
    start_time = time.time()
    trainer_base = train_model(
        model_base,
        train_loader,
        device,
        config,
        use_adnlcc=False,
        num_epochs=num_epochs
    )
    base_training_time = time.time() - start_time
    print(f"Base model training completed in {base_training_time:.2f} seconds")
    
    print("\n" + "-"*80)
    print("Training ADNLCC Model...")
    print("-"*80)
    start_time = time.time()
    trainer_adnlcc = train_model(
        model_adnlcc,
        train_loader,
        device,
        config,
        use_adnlcc=True,
        num_epochs=num_epochs
    )
    adnlcc_training_time = time.time() - start_time
    print(f"ADNLCC model training completed in {adnlcc_training_time:.2f} seconds")
    
    print("\n" + "-"*80)
    print("Evaluating Models...")
    print("-"*80)
    results = evaluate_model(
        model_base,
        model_adnlcc,
        test_loader,
        test_loader.dataset,
        device,
        config
    )
    
    print("\n" + "="*80)
    print("Experiment Results:")
    print("="*80)
    print(f"FID Score (Base Method): {results['fid_base']:.4f}")
    print(f"FID Score (ADNLCC): {results['fid_adnlcc']:.4f}")
    print(f"Average SSIM (Base Method): {results['ssim_base']:.4f}")
    print(f"Average SSIM (ADNLCC): {results['ssim_adnlcc']:.4f}")
    
    print("\nGenerated Figures:")
    for name, path in results['sample_paths'].items():
        print(f"  {name}: {path}")
    
    print("\nTraining Time Comparison:")
    print(f"  Base Method: {base_training_time:.2f} seconds")
    print(f"  ADNLCC: {adnlcc_training_time:.2f} seconds")
    print(f"  Speedup: {base_training_time / adnlcc_training_time:.2f}x")
    
    print("\n" + "="*80)
    print(f"ADNLCC experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return results

def main():
    """Main function to parse arguments and run the experiment."""
    parser = argparse.ArgumentParser(description='Run ADNLCC experiment')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--noise_level', type=float, default=0.5, help='Noise level for training')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda or cpu)')
    
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with minimal epochs')
    
    args = parser.parse_args()
    
    config = create_config(args)
    config_path = save_config(config)
    print(f"Configuration saved to: {config_path}")
    
    results = run_experiment(config)
    
    return results

if __name__ == "__main__":
    main()
