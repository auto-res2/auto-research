#!/usr/bin/env python3
"""
Main script for running CGCD experiments.

This script orchestrates the entire experimental pipeline:
1. Performance Comparison on Noisy and Complex Data Distributions
2. Ablation Study of Adaptive Components
3. Experiments on Continuous Feature Domains with External Cues

It uses the modules from preprocess.py, train.py, and evaluate.py to implement
the complete workflow from data preprocessing to model evaluation.
"""

import os
import argparse
import torch
import torch.optim as optim
import time
import json

from preprocess import (
    get_cifar10_dataloader, 
    get_continuous_dataloader, 
    get_condition
)
from train import (
    BaseDiffusionModel, 
    CGCDDiffusionModel, 
    DiffusionModelVariant, 
    ContinuousCGCDModel,
    train_experiment1,
    train_experiment2,
    train_experiment3
)
from evaluate import (
    evaluate_experiment1,
    evaluate_experiment2,
    evaluate_experiment3
)
from utils.diffusion_utils import (
    seed_everything, 
    get_device, 
    save_model, 
    load_model
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CGCD experiments')
    
    parser.add_argument('--config', type=str, default='config/default_config.json',
                        help='Path to configuration file')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with minimal computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches per epoch (for testing)')
    parser.add_argument('--experiment', type=int, default=0,
                        help='Run specific experiment (0=all, 1/2/3=specific experiment)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only (no training)')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Dictionary of configuration parameters
    """
    # Default configuration
    default_config = {
        'seed': 42,
        'batch_size': 64,
        'epochs': 5,
        'lr': 1e-3,
        'max_batches': None,
        'test_mode': False,
        'eval_only': False,
        'experiment': 0,
        'log_dir': 'logs',
        'model_dir': 'models',
        'data_dir': 'data'
    }
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # If config file exists, load it
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Update default config with loaded config
        default_config.update(config)
        print(f"Configuration loaded from {config_path}")
    else:
        print(f"Configuration file {config_path} not found. Using default configuration.")
        # Save default config for future use
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Default configuration saved to {config_path}")
    
    return default_config

def update_config_with_args(config, args):
    """
    Update configuration with command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        config: Updated configuration dictionary
    """
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    return config

def setup_directories(config):
    """
    Create necessary directories.
    
    Args:
        config: Configuration dictionary
    """
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['data_dir'], exist_ok=True)
    
    print(f"Directories created: {config['log_dir']}, {config['model_dir']}, {config['data_dir']}")

def run_experiment1(config, device, trainloader, testloader):
    """
    Run Experiment 1: Performance Comparison on Noisy and Complex Data Distributions.
    
    Args:
        config: Configuration dictionary
        device: Device to run on
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        
    Returns:
        base_model: Trained base model
        cgcd_model: Trained CGCD model
        exp1_results: Experiment 1 results
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: PERFORMANCE COMPARISON ON NOISY AND COMPLEX DATA DISTRIBUTIONS")
    print("="*70)
    
    # Initialize models
    base_model = BaseDiffusionModel().to(device)
    cgcd_model = CGCDDiffusionModel().to(device)
    
    # Initialize optimizers
    base_optimizer = optim.Adam(base_model.parameters(), lr=config['lr'])
    cgcd_optimizer = optim.Adam(cgcd_model.parameters(), lr=config['lr'])
    
    if not config['eval_only']:
        # Train models
        print("\nTraining Base Model...")
        base_model = train_experiment1(
            base_model, base_optimizer, trainloader, model_name='Base',
            epochs=config['epochs'], max_batches=config['max_batches'],
            device=device, log_dir=config['log_dir'], model_dir=config['model_dir'],
            test_mode=config['test_mode']
        )
        
        print("\nTraining CGCD Model...")
        cgcd_model = train_experiment1(
            cgcd_model, cgcd_optimizer, trainloader, model_name='CGCD',
            epochs=config['epochs'], max_batches=config['max_batches'],
            device=device, log_dir=config['log_dir'], model_dir=config['model_dir'],
            test_mode=config['test_mode']
        )
        
        # Save models
        save_model(base_model, config['model_dir'], "Base_final.pt")
        save_model(cgcd_model, config['model_dir'], "CGCD_final.pt")
    else:
        # Load models if available
        try:
            base_model = load_model(base_model, config['model_dir'], "Base_final.pt")
            cgcd_model = load_model(cgcd_model, config['model_dir'], "CGCD_final.pt")
        except:
            print("Warning: Could not load trained models. Using untrained models for evaluation.")
    
    # Evaluate models
    print("\nEvaluating models...")
    exp1_results = evaluate_experiment1(
        base_model, cgcd_model, testloader,
        max_batches=config['max_batches'], device=device, log_dir=config['log_dir']
    )
    
    return base_model, cgcd_model, exp1_results

def run_experiment2(config, device, trainloader, testloader):
    """
    Run Experiment 2: Ablation Study of Adaptive Components.
    
    Args:
        config: Configuration dictionary
        device: Device to run on
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        
    Returns:
        variants: List of trained model variants
        exp2_results: Experiment 2 results
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: ABLATION STUDY OF ADAPTIVE COMPONENTS")
    print("="*70)
    
    # Initialize model variants
    variant_full = DiffusionModelVariant(adaptive_noise=True, soft_assimilation=True).to(device)
    variant_fixed = DiffusionModelVariant(adaptive_noise=False, soft_assimilation=True).to(device)
    variant_hard = DiffusionModelVariant(adaptive_noise=True, soft_assimilation=False).to(device)
    
    # Initialize optimizers
    optimizer_full = optim.Adam(variant_full.parameters(), lr=config['lr'])
    optimizer_fixed = optim.Adam(variant_fixed.parameters(), lr=config['lr'])
    optimizer_hard = optim.Adam(variant_hard.parameters(), lr=config['lr'])
    
    if not config['eval_only']:
        # Train model variants
        print("\nTraining Full Variant (Adaptive Noise + Soft Assimilation)...")
        variant_full = train_experiment2(
            variant_full, optimizer_full, trainloader, variant_name='Variant_Full',
            epochs=config['epochs'], max_batches=config['max_batches'],
            device=device, log_dir=config['log_dir'], model_dir=config['model_dir'],
            test_mode=config['test_mode']
        )
        
        print("\nTraining Fixed Noise Variant (Fixed Noise + Soft Assimilation)...")
        variant_fixed = train_experiment2(
            variant_fixed, optimizer_fixed, trainloader, variant_name='Variant_Fixed',
            epochs=config['epochs'], max_batches=config['max_batches'],
            device=device, log_dir=config['log_dir'], model_dir=config['model_dir'],
            test_mode=config['test_mode']
        )
        
        print("\nTraining Hard Conditioning Variant (Adaptive Noise + Hard Conditioning)...")
        variant_hard = train_experiment2(
            variant_hard, optimizer_hard, trainloader, variant_name='Variant_Hard',
            epochs=config['epochs'], max_batches=config['max_batches'],
            device=device, log_dir=config['log_dir'], model_dir=config['model_dir'],
            test_mode=config['test_mode']
        )
        
        # Save models
        save_model(variant_full, config['model_dir'], "Variant_Full_final.pt")
        save_model(variant_fixed, config['model_dir'], "Variant_Fixed_final.pt")
        save_model(variant_hard, config['model_dir'], "Variant_Hard_final.pt")
    else:
        # Load models if available
        try:
            variant_full = load_model(variant_full, config['model_dir'], "Variant_Full_final.pt")
            variant_fixed = load_model(variant_fixed, config['model_dir'], "Variant_Fixed_final.pt")
            variant_hard = load_model(variant_hard, config['model_dir'], "Variant_Hard_final.pt")
        except:
            print("Warning: Could not load trained models. Using untrained models for evaluation.")
    
    # Evaluate model variants
    print("\nEvaluating model variants...")
    variants = [variant_full, variant_fixed, variant_hard]
    variant_names = ['Variant_Full', 'Variant_Fixed', 'Variant_Hard']
    
    exp2_results = evaluate_experiment2(
        variants, variant_names, testloader,
        max_batches=config['max_batches'], device=device, log_dir=config['log_dir']
    )
    
    return variants, exp2_results

def run_experiment3(config, device, continuous_loader):
    """
    Run Experiment 3: Continuous Feature Domains with External Cues.
    
    Args:
        config: Configuration dictionary
        device: Device to run on
        continuous_loader: DataLoader for continuous data
        
    Returns:
        continuous_model: Trained continuous model
        exp3_results: Experiment 3 results
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: CONTINUOUS FEATURE DOMAINS WITH EXTERNAL CUES")
    print("="*70)
    
    # Initialize continuous model
    continuous_model = ContinuousCGCDModel().to(device)
    
    # Initialize optimizer
    continuous_optimizer = optim.Adam(continuous_model.parameters(), lr=config['lr'])
    
    if not config['eval_only']:
        # Train continuous model
        print("\nTraining Continuous CGCD Model...")
        continuous_model = train_experiment3(
            continuous_model, continuous_optimizer, continuous_loader,
            epochs=config['epochs'], max_batches=config['max_batches'],
            device=device, log_dir=config['log_dir'], model_dir=config['model_dir'],
            test_mode=config['test_mode']
        )
        
        # Save model
        save_model(continuous_model, config['model_dir'], "Continuous_final.pt")
    else:
        # Load model if available
        try:
            continuous_model = load_model(continuous_model, config['model_dir'], "Continuous_final.pt")
        except:
            print("Warning: Could not load trained model. Using untrained model for evaluation.")
    
    # Evaluate continuous model
    print("\nEvaluating continuous model...")
    exp3_results = evaluate_experiment3(
        continuous_model, continuous_loader,
        max_batches=config['max_batches'], device=device, log_dir=config['log_dir']
    )
    
    return continuous_model, exp3_results

def test_code(device):
    """
    Run a quick test to ensure the code works.
    
    Args:
        device: Device to run on
    """
    print("\n" + "="*70)
    print("RUNNING QUICK TEST")
    print("="*70)
    
    # Get a small batch of data
    from preprocess import get_cifar10_dataloader, get_continuous_dataloader
    trainloader, _ = get_cifar10_dataloader(batch_size=8)
    data, labels = next(iter(trainloader))
    data, labels = data.to(device), labels.to(device)
    condition = get_condition(labels).to(device)
    
    # Test BaseDiffusionModel
    print("\nTesting BaseDiffusionModel...")
    base_model = BaseDiffusionModel().to(device)
    noise_level = 0.2
    alpha_t = torch.tensor(1.0).to(device)
    base_output = base_model(data, condition, noise_level, alpha_t)
    print(f"Base Model output shape: {base_output.shape}")
    
    # Test CGCDDiffusionModel
    print("\nTesting CGCDDiffusionModel...")
    cgcd_model = CGCDDiffusionModel().to(device)
    cgcd_output, dyn_alpha = cgcd_model(data, condition, noise_level, alpha_t)
    print(f"CGCD Model output shape: {cgcd_output.shape}")
    print(f"Dynamic Alpha (mean): {dyn_alpha.mean().item():.4f}")
    
    # Test model variants
    print("\nTesting Model Variants...")
    variant_full = DiffusionModelVariant(adaptive_noise=True, soft_assimilation=True).to(device)
    variant_fixed = DiffusionModelVariant(adaptive_noise=False, soft_assimilation=True).to(device)
    variant_hard = DiffusionModelVariant(adaptive_noise=True, soft_assimilation=False).to(device)
    
    out_full, alpha_full = variant_full(data, condition, noise_level, alpha_t)
    out_fixed, alpha_fixed = variant_fixed(data, condition, noise_level, alpha_t)
    out_hard, alpha_hard = variant_hard(data, condition, noise_level, alpha_t)
    
    print(f"Variant Full output shape: {out_full.shape}, alpha: {alpha_full.mean().item():.4f}")
    print(f"Variant Fixed Noise output shape: {out_fixed.shape}, alpha: {alpha_fixed.mean().item():.4f}")
    print(f"Variant Hard Conditioning output shape: {out_hard.shape}, alpha: {alpha_hard.mean().item():.4f}")
    
    # Test ContinuousCGCDModel
    print("\nTesting ContinuousCGCDModel...")
    continuous_model = ContinuousCGCDModel().to(device)
    # Create dummy continuous signals in [0,1]
    dummy_signal = torch.rand(8).to(device)
    cont_out, cont_dyn_alpha = continuous_model(data, dummy_signal, noise_level, alpha_t)
    print(f"Continuous Model output shape: {cont_out.shape}")
    print(f"Continuous Dynamic Alpha (mean): {cont_dyn_alpha.mean().item():.4f}")
    
    print("\nQuick test completed successfully!")

def main():
    """Main function to run CGCD experiments."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set random seed for reproducibility
    seed_everything(config['seed'])
    
    # Get device
    device = get_device()
    
    # Create necessary directories
    setup_directories(config)
    
    # Check if we're just running a quick test
    if config['test_mode'] and config['max_batches'] is None:
        config['max_batches'] = 3
        config['epochs'] = 1
    
    # Load data
    print("\nLoading data...")
    trainloader, testloader = get_cifar10_dataloader(
        batch_size=config['batch_size'],
        data_dir=config['data_dir']
    )
    continuous_loader = get_continuous_dataloader(
        batch_size=config['batch_size'],
        n_samples=500
    )
    
    # Run quick test if in test mode
    if config['test_mode']:
        test_code(device)
    
    # Run experiments
    results = {}
    
    if config['experiment'] == 0 or config['experiment'] == 1:
        # Run Experiment 1
        base_model, cgcd_model, exp1_results = run_experiment1(
            config, device, trainloader, testloader
        )
        results['experiment1'] = exp1_results
    
    if config['experiment'] == 0 or config['experiment'] == 2:
        # Run Experiment 2
        variants, exp2_results = run_experiment2(
            config, device, trainloader, testloader
        )
        results['experiment2'] = exp2_results
    
    if config['experiment'] == 0 or config['experiment'] == 3:
        # Run Experiment 3
        continuous_model, exp3_results = run_experiment3(
            config, device, continuous_loader
        )
        results['experiment3'] = exp3_results
    
    # Print overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    if 'experiment1' in results:
        print("\nExperiment 1 (Performance Comparison):")
        print(f"Base Model MSE: {results['experiment1']['base_metrics']['mse']:.4f}")
        print(f"CGCD Model MSE: {results['experiment1']['cgcd_metrics']['mse']:.4f}")
        print(f"Improvement: {results['experiment1']['mse_improvement']:.4f} "
              f"({results['experiment1']['mse_improvement']/results['experiment1']['base_metrics']['mse']*100:.2f}%)")
    
    if 'experiment2' in results:
        print("\nExperiment 2 (Ablation Study):")
        for name, metrics in results['experiment2'].items():
            print(f"{name}: MSE = {metrics['mse']:.4f}")
    
    if 'experiment3' in results:
        print("\nExperiment 3 (Continuous Features):")
        print(f"MSE: {results['experiment3']['mse']:.4f}")
        print(f"Signal-Latent Correlation: {results['experiment3']['correlation']:.4f}")
    
    print("\nExperiments completed successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
