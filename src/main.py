#!/usr/bin/env python3
import os
import json
import torch
import argparse
import numpy as np
import random
from datetime import datetime

# Import modules from the project
from preprocess import preprocess_data
from train import train_experiment, set_seeds, Generator, ScoreNetworkABSD, ScoreNetworkSiD
from evaluate import evaluate_model, compare_models, evaluate_sensitivity_results

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['./data', './models', './logs', './results', './config']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_config(config, filename='./config/experiment_config.json'):
    """Save configuration to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filename}")

def load_config(filename='./config/experiment_config.json'):
    """Load configuration from a JSON file if it exists, otherwise return default config."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {filename}")
        return config
    else:
        print(f"Configuration file {filename} not found. Using default configuration.")
        return get_default_config()

def get_default_config():
    """Return default configuration for experiments."""
    return {
        'num_epochs': 2,
        'batch_size': 64,
        'image_size': 32,
        'latent_dim': 100,
        'image_channels': 3,
        'learning_rate': 1e-4,
        'uncertainty_factor': 1.0,
        'sde_stepsize': 0.005,
        'uncertainty_factors': [0.1, 0.5, 1.0, 2.0],
        'sde_stepsizes': [0.001, 0.005, 0.01],
        'data_dir': './data',
        'seed': 42
    }

def run_performance_experiment(config, device, log_dir='./logs'):
    """Run the performance benchmarking experiment."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: PERFORMANCE BENCHMARKING")
    print("="*80)
    
    print("Loading and preprocessing data...")
    dataloader = preprocess_data(config)
    
    print("Running performance experiment...")
    results = train_experiment('performance', dataloader, config, device, log_dir)
    
    # Load trained models for evaluation
    print("Loading trained models for evaluation...")
    generator_absd = Generator(
        config['latent_dim'], 
        config['image_channels'], 
        config['image_size']
    ).to(device)
    generator_sid = Generator(
        config['latent_dim'], 
        config['image_channels'], 
        config['image_size']
    ).to(device)
    
    generator_absd.load_state_dict(torch.load('./models/generator_absd.pth'))
    generator_sid.load_state_dict(torch.load('./models/generator_sid.pth'))
    
    # Compare models
    print("Evaluating and comparing models...")
    models_dict = {
        'ABSD': generator_absd,
        'SiD': generator_sid
    }
    
    evaluation_results = compare_models(
        models_dict, 
        dataloader, 
        device, 
        num_samples=64, 
        save_dir='./results/performance'
    )
    
    # Save results
    os.makedirs('./results/performance', exist_ok=True)
    with open('./results/performance/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nPerformance experiment completed.")
    print("Results saved to ./results/performance/")
    
    # Print summary of results
    print("\nSUMMARY OF PERFORMANCE EXPERIMENT RESULTS:")
    print("-"*50)
    for model_name, metrics in evaluation_results.items():
        print(f"{model_name}:")
        print(f"  FID Score: {metrics['fid']:.2f}")
        print(f"  Mean Pixel Value: {metrics['mean']:.4f}")
        print(f"  Pixel Value Std: {metrics['std']:.4f}")
    
    return results, evaluation_results

def run_ablation_experiment(config, device, log_dir='./logs'):
    """Run the ablation study experiment."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: ABLATION STUDY")
    print("="*80)
    
    print("Loading and preprocessing data...")
    dataloader = preprocess_data(config)
    
    print("Running ablation experiment...")
    results = train_experiment('ablation', dataloader, config, device, log_dir)
    
    # Load trained models for evaluation
    print("Loading trained model for evaluation...")
    generator_model = Generator(
        config['latent_dim'], 
        config['image_channels'], 
        config['image_size']
    ).to(device)
    
    generator_model.load_state_dict(torch.load('./models/generator_ablation.pth'))
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(
        generator_model, 
        dataloader, 
        device, 
        num_samples=64, 
        save_dir='./results/ablation'
    )
    
    # Save results
    os.makedirs('./results/ablation', exist_ok=True)
    with open('./results/ablation/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAblation experiment completed.")
    print("Results saved to ./results/ablation/")
    
    # Print summary of results
    print("\nSUMMARY OF ABLATION EXPERIMENT RESULTS:")
    print("-"*50)
    print("Full ABSD vs. Control Variant:")
    
    for variant, epochs in results.items():
        last_epoch = epochs[-1]
        print(f"{variant}:")
        print(f"  Final Loss: {last_epoch['loss']:.4f}")
        print(f"  Final Gradient Norm: {last_epoch['grad_norm']:.4f}")
    
    return results, metrics

def run_sensitivity_experiment(config, device, log_dir='./logs'):
    """Run the sensitivity analysis experiment."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: SENSITIVITY ANALYSIS")
    print("="*80)
    
    print("Loading and preprocessing data...")
    dataloader = preprocess_data(config)
    
    print("Running sensitivity analysis experiment...")
    results = train_experiment('sensitivity', dataloader, config, device, log_dir)
    
    # Analyze sensitivity results
    print("Analyzing sensitivity results...")
    sensitivity_analysis = evaluate_sensitivity_results(
        './logs/sensitivity_results.json', 
        save_dir='./results/sensitivity'
    )
    
    print("\nSensitivity analysis experiment completed.")
    print("Results saved to ./results/sensitivity/")
    
    # Print summary of results
    print("\nSUMMARY OF SENSITIVITY ANALYSIS RESULTS:")
    print("-"*50)
    
    # Find best hyperparameters based on FID
    best_uf = None
    best_ss = None
    best_fid = float('inf')
    
    for uf in sensitivity_analysis['uncertainty_factors']:
        for ss in sensitivity_analysis['sde_stepsizes']:
            fid = sensitivity_analysis['fid_values'][uf][ss]
            if fid < best_fid:
                best_fid = fid
                best_uf = uf
                best_ss = ss
    
    print(f"Best hyperparameters found:")
    print(f"  Uncertainty Factor: {best_uf}")
    print(f"  SDE Stepsize: {best_ss}")
    print(f"  FID Score: {best_fid:.2f}")
    
    return results, sensitivity_analysis

def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description='Run ABSD experiments')
    parser.add_argument('--config', type=str, default='./config/experiment_config.json',
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'performance', 'ablation', 'sensitivity'],
                        help='Which experiment to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed != 42:
        config['seed'] = args.seed
    
    # Set random seeds for reproducibility
    set_seeds(config['seed'])
    
    # Save configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_config(config, f'./config/experiment_config_{timestamp}.json')
    
    # Create log directory with timestamp
    log_dir = f'./logs/{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Run experiments
    results = {}
    
    if args.experiment in ['all', 'performance']:
        print("\nRunning Performance Benchmarking Experiment...")
        results['performance'] = run_performance_experiment(config, device, log_dir)
    
    if args.experiment in ['all', 'ablation']:
        print("\nRunning Ablation Study Experiment...")
        results['ablation'] = run_ablation_experiment(config, device, log_dir)
    
    if args.experiment in ['all', 'sensitivity']:
        print("\nRunning Sensitivity Analysis Experiment...")
        results['sensitivity'] = run_sensitivity_experiment(config, device, log_dir)
    
    # Save overall results summary
    with open(f'./results/experiment_summary_{timestamp}.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': config,
            'device': str(device),
            'experiments_run': args.experiment
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to ./results/")
    print(f"Logs saved to {log_dir}/")

if __name__ == "__main__":
    main()
