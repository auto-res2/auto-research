"""
Main module for running ACM optimizer experiments.

This script orchestrates the entire experimental pipeline:
1. Synthetic function benchmarking
2. CIFAR-10 image classification
3. Ablation studies

The experiments compare the novel Adaptive Curvature Momentum (ACM) optimizer
with existing optimizers and ablated variants.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from datetime import datetime

# Import modules
from preprocess import create_synthetic_data, load_cifar10, get_subset_loaders
from train import SimpleCNN, train_synthetic, train_cifar10, run_ablation_study
from evaluate import (
    evaluate_synthetic_results, 
    evaluate_cifar10_model, 
    compare_optimizers, 
    evaluate_ablation_results
)
from utils.optimizers import ACM, ACM_NoCurvature, ACM_NoRegularization

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['./logs', './models', './data', './paper']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories setup complete.")

def experiment_1_synthetic_benchmarking(config):
    """
    Experiment 1: Synthetic Function Benchmarking
    
    Compare ACM with other optimizers on synthetic optimization problems.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: SYNTHETIC FUNCTION BENCHMARKING")
    print("="*80)
    
    # Setup optimizers to compare
    optimizers = {
        'ACM': (ACM, {
            'lr': config['synthetic']['lr'], 
            'betas': (config['synthetic']['beta1'], config['synthetic']['beta2']), 
            'curvature_coeff': config['synthetic']['curvature_coeff']
        }),
        'Adam': (optim.Adam, {'lr': config['synthetic']['lr']}),
        'SGD': (optim.SGD, {'lr': config['synthetic']['lr'], 'momentum': config['synthetic']['beta1']})
    }
    
    # Run Rosenbrock function optimization
    print("\nRunning Rosenbrock function optimization...")
    rosenbrock_func, rosenbrock_init = create_synthetic_data('rosenbrock')
    rosenbrock_results = {}
    
    for name, (opt_class, kwargs) in optimizers.items():
        print(f"\nOptimizing Rosenbrock with {name}")
        traj, losses = train_synthetic(
            rosenbrock_func, 
            rosenbrock_init, 
            opt_class, 
            kwargs,
            num_iterations=config['synthetic']['iterations'],
            loss_threshold=config['synthetic']['loss_threshold']
        )
        rosenbrock_results[name] = {'trajectory': traj, 'losses': losses}
    
    # Evaluate Rosenbrock results
    rosenbrock_metrics = evaluate_synthetic_results(rosenbrock_results, 'rosenbrock')
    
    # Run ill-conditioned quadratic function optimization
    print("\nRunning ill-conditioned quadratic function optimization...")
    quadratic_func, quadratic_init = create_synthetic_data('ill_conditioned')
    quadratic_results = {}
    
    for name, (opt_class, kwargs) in optimizers.items():
        print(f"\nOptimizing ill-conditioned quadratic with {name}")
        traj, losses = train_synthetic(
            quadratic_func, 
            quadratic_init, 
            opt_class, 
            kwargs,
            num_iterations=config['synthetic']['iterations'],
            loss_threshold=config['synthetic']['loss_threshold']
        )
        quadratic_results[name] = {'trajectory': traj, 'losses': losses}
    
    # Evaluate quadratic results
    quadratic_metrics = evaluate_synthetic_results(quadratic_results, 'ill_conditioned')
    
    return {
        'rosenbrock': {'results': rosenbrock_results, 'metrics': rosenbrock_metrics},
        'quadratic': {'results': quadratic_results, 'metrics': quadratic_metrics}
    }

def experiment_2_cifar10_classification(config):
    """
    Experiment 2: CIFAR-10 Image Classification
    
    Compare ACM with other optimizers on CIFAR-10 image classification.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: CIFAR-10 IMAGE CLASSIFICATION")
    print("="*80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader, classes = load_cifar10(
        batch_size=config['cifar10']['batch_size'],
        num_workers=config['cifar10']['num_workers']
    )
    
    # For quick testing, use subset of data if specified
    if config['cifar10']['use_subset']:
        print(f"Using subset of data ({config['cifar10']['subset_batches']} batches)")
        train_subset, val_subset = get_subset_loaders(
            train_loader, val_loader, 
            num_batches=config['cifar10']['subset_batches']
        )
        train_loader, val_loader = train_subset, val_subset
    
    # Setup optimizers to compare
    optimizers = {
        'ACM': (ACM, {
            'lr': config['cifar10']['lr'], 
            'betas': (config['cifar10']['beta1'], config['cifar10']['beta2']), 
            'curvature_coeff': config['cifar10']['curvature_coeff'],
            'weight_decay': config['cifar10']['weight_decay']
        }),
        'Adam': (optim.Adam, {
            'lr': config['cifar10']['lr'],
            'weight_decay': config['cifar10']['weight_decay']
        }),
        'SGD': (optim.SGD, {
            'lr': config['cifar10']['lr'], 
            'momentum': config['cifar10']['beta1'],
            'weight_decay': config['cifar10']['weight_decay']
        })
    }
    
    # Train models with different optimizers
    histories = {}
    models = {}
    
    for name, (opt_class, kwargs) in optimizers.items():
        print(f"\nTraining CIFAR-10 model with {name}")
        model = SimpleCNN()
        history = train_cifar10(
            model, 
            train_loader, 
            val_loader, 
            opt_class, 
            kwargs,
            num_epochs=config['cifar10']['epochs'],
            device=device,
            experiment_name=f'cifar10_{name.lower()}'
        )
        histories[name] = history
        models[name] = model
    
    # Compare optimizer performance
    comparison_metrics = compare_optimizers(histories)
    
    # Evaluate best model on test set
    best_optimizer = max(comparison_metrics.items(), key=lambda x: x[1]['final_val_acc'])[0]
    print(f"\nBest optimizer: {best_optimizer}")
    print(f"Evaluating {best_optimizer} model on test set...")
    
    # Load fresh test loader for final evaluation
    _, test_loader, _ = load_cifar10(
        batch_size=config['cifar10']['batch_size'],
        num_workers=config['cifar10']['num_workers']
    )
    
    best_model_metrics = evaluate_cifar10_model(models[best_optimizer], test_loader, device)
    
    return {
        'histories': histories,
        'comparison_metrics': comparison_metrics,
        'best_optimizer': best_optimizer,
        'best_model_metrics': best_model_metrics
    }

def experiment_3_ablation_study(config):
    """
    Experiment 3: Ablation Studies for ACM Components
    
    Compare different variants of ACM to understand the contribution of each component.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: ABLATION STUDIES FOR ACM COMPONENTS")
    print("="*80)
    
    # Run ablation study on ill-conditioned quadratic function
    print("\nRunning ablation study on ill-conditioned quadratic function...")
    quadratic_func, quadratic_init = create_synthetic_data('ill_conditioned')
    
    ablation_results = run_ablation_study(
        quadratic_func, 
        quadratic_init,
        num_iterations=config['ablation']['iterations'],
        loss_threshold=config['ablation']['loss_threshold']
    )
    
    # Evaluate ablation results
    ablation_metrics = evaluate_ablation_results(ablation_results)
    
    return {
        'results': ablation_results,
        'metrics': ablation_metrics
    }

def load_config():
    """Load configuration for experiments."""
    # Default configuration
    config = {
        'synthetic': {
            'iterations': 5000,
            'loss_threshold': 1e-3,
            'lr': 1e-3,
            'beta1': 0.9,
            'beta2': 0.999,
            'curvature_coeff': 1e-2
        },
        'cifar10': {
            'epochs': 10,
            'batch_size': 128,
            'num_workers': 2,
            'lr': 1e-3,
            'beta1': 0.9,
            'beta2': 0.999,
            'curvature_coeff': 1e-2,
            'weight_decay': 1e-4,
            'use_subset': False,
            'subset_batches': 10
        },
        'ablation': {
            'iterations': 2000,
            'loss_threshold': 1e-5
        }
    }
    
    # Check if config file exists
    config_path = './config/experiment_config.py'
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            if spec is not None and spec.loader is not None:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                # Update config with values from file
                if hasattr(config_module, 'config'):
                    for section, params in config_module.config.items():
                        if section in config:
                            config[section].update(params)
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    return config

def main():
    """Main function to run all experiments."""
    start_time = time.time()
    
    # Setup directories
    setup_directories()
    
    # Load configuration
    config = load_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ACM optimizer experiments')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3], help='Run specific experiment (1, 2, or 3)')
    parser.add_argument('--quick', action='store_true', help='Run quick test with reduced iterations/epochs')
    args = parser.parse_args()
    
    # Apply quick test settings if requested
    if args.quick:
        print("Running quick test with reduced iterations/epochs")
        config['synthetic']['iterations'] = 100
        config['synthetic']['loss_threshold'] = 1e-1
        config['cifar10']['epochs'] = 1
        config['cifar10']['use_subset'] = True
        config['cifar10']['subset_batches'] = 5
        config['ablation']['iterations'] = 100
        config['ablation']['loss_threshold'] = 1e-1
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    for section, params in config.items():
        print(f"  {section}:")
        for param, value in params.items():
            print(f"    {param}: {value}")
    
    # Run experiments
    results = {}
    
    if args.exp is None or args.exp == 1:
        results['experiment_1'] = experiment_1_synthetic_benchmarking(config)
    
    if args.exp is None or args.exp == 2:
        results['experiment_2'] = experiment_2_cifar10_classification(config)
    
    if args.exp is None or args.exp == 3:
        results['experiment_3'] = experiment_3_ablation_study(config)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    if 'experiment_1' in results:
        print("\nExperiment 1 (Synthetic Benchmarking):")
        for func, data in results['experiment_1'].items():
            print(f"  {func.capitalize()} function:")
            for optimizer, metrics in data['metrics'].items():
                print(f"    {optimizer}: Final loss = {metrics['final_loss']:.4e}, "
                      f"Iterations = {metrics['iterations']}")
    
    if 'experiment_2' in results:
        print("\nExperiment 2 (CIFAR-10 Classification):")
        best_opt = results['experiment_2']['best_optimizer']
        best_acc = results['experiment_2']['comparison_metrics'][best_opt]['final_val_acc']
        print(f"  Best optimizer: {best_opt} (Validation accuracy: {best_acc:.4f})")
        if 'best_model_metrics' in results['experiment_2']:
            test_acc = results['experiment_2']['best_model_metrics']['accuracy']
            print(f"  Test accuracy: {test_acc:.4f}%")
    
    if 'experiment_3' in results:
        print("\nExperiment 3 (Ablation Study):")
        for variant, metrics in results['experiment_3']['metrics'].items():
            print(f"  {variant}: Final loss = {metrics['final_loss']:.4e}, "
                  f"Iterations = {metrics['iterations']}")
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()
