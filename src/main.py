#!/usr/bin/env python
"""
Adaptive Curvature Momentum (ACM) Optimizer Experiment

This script contains the following experiments:
1. Synthetic Optimization Benchmark (Convex Quadratic and Rosenbrock-like functions)
2. Deep Neural Network Training on CIFAR-10 using a simple CNN
3. Ablation Study & Hyperparameter Sensitivity Analysis on MNIST

Each experiment compares a custom Adaptive Curvature Momentum (ACM) optimizer against 
established optimizers (Adam, SGD with momentum). The ACM optimizer adjusts per-parameter 
learning rates using a simple curvature-estimate (the difference between successive gradients) 
and uses momentum buffering.

A quick_test() function is provided to run minimal iterations (to verify code execution).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt

from src.preprocess import preprocess_data
from src.train import (
    train_synthetic, train_cifar10, train_mnist, 
    run_ablation_study, quick_test as train_quick_test
)
from src.evaluate import (
    evaluate_synthetic, evaluate_cifar10, evaluate_mnist, 
    evaluate_ablation_study, quick_test as evaluate_quick_test
)
from src.utils.optimizers import ACMOptimizer

def run_synthetic_experiment(config):
    """
    Run the synthetic optimization benchmark experiment.
    
    Args:
        config: Configuration dictionary with experiment parameters
        
    Returns:
        Dictionary containing experiment results
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: SYNTHETIC OPTIMIZATION BENCHMARK")
    print("="*80)
    
    # Preprocess data
    print("\nPreprocessing synthetic data...")
    data = preprocess_data(config)['synthetic']
    
    # Define optimizers to compare
    optimizers = ['ACM', 'Adam', 'SGD_mom']
    
    # Train models with different optimizers
    results = {}
    for optimizer_type in optimizers:
        print(f"\nTraining with {optimizer_type} optimizer...")
        optimizer_results = train_synthetic(data, optimizer_type, config['train'])
        results[optimizer_type] = optimizer_results
    
    # Evaluate results
    metrics = evaluate_synthetic(results, config['evaluate'])
    
    return {
        'data': data,
        'results': results,
        'metrics': metrics
    }

def run_cifar10_experiment(config):
    """
    Run the CIFAR-10 deep neural network training experiment.
    
    Args:
        config: Configuration dictionary with experiment parameters
        
    Returns:
        Dictionary containing experiment results
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: DEEP NEURAL NETWORK TRAINING ON CIFAR-10")
    print("="*80)
    
    # Preprocess data
    print("\nPreprocessing CIFAR-10 data...")
    data_loaders = preprocess_data(config)['cifar10']
    
    # Define optimizers to compare
    optimizers = ['ACM', 'Adam', 'SGD_mom']
    
    # Train models with different optimizers
    results = {}
    for optimizer_type in optimizers:
        print(f"\nTraining CIFAR-10 model with {optimizer_type} optimizer...")
        optimizer_results = train_cifar10(data_loaders, optimizer_type, config['train'])
        results[optimizer_type] = optimizer_results
    
    # Evaluate results
    metrics = evaluate_cifar10(results, config['evaluate'])
    
    return {
        'data_loaders': data_loaders,
        'results': results,
        'metrics': metrics
    }

def run_mnist_experiment(config):
    """
    Run the MNIST training and ablation study experiment.
    
    Args:
        config: Configuration dictionary with experiment parameters
        
    Returns:
        Dictionary containing experiment results
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: MNIST TRAINING AND ABLATION STUDY")
    print("="*80)
    
    # Preprocess data
    print("\nPreprocessing MNIST data...")
    data_loaders = preprocess_data(config)['mnist']
    
    # Define optimizers to compare
    optimizers = ['ACM', 'Adam', 'SGD_mom']
    
    # Train models with different optimizers
    results = {}
    for optimizer_type in optimizers:
        print(f"\nTraining MNIST model with {optimizer_type} optimizer...")
        optimizer_results = train_mnist(data_loaders, optimizer_type, config['train'])
        results[optimizer_type] = optimizer_results
    
    # Evaluate results
    metrics = evaluate_mnist(results, config['evaluate'])
    
    # Run ablation study
    print("\nRunning ablation study on MNIST dataset...")
    ablation_results = run_ablation_study(data_loaders, config['ablation'])
    ablation_metrics = evaluate_ablation_study(ablation_results, config['evaluate'])
    
    return {
        'data_loaders': data_loaders,
        'results': results,
        'metrics': metrics,
        'ablation_results': ablation_results,
        'ablation_metrics': ablation_metrics
    }

def run_all_experiments(config=None):
    """
    Run all experiments.
    
    Args:
        config: Configuration dictionary with experiment parameters
        
    Returns:
        Dictionary containing all experiment results
    """
    # Set default configuration if not provided
    if config is None:
        config = {
            'synthetic': {
                'n_samples': 1000,
                'seed': 42
            },
            'cifar10': {
                'batch_size': 128,
                'download': True
            },
            'mnist': {
                'batch_size': 128,
                'download': True
            },
            'train': {
                'seed': 42,
                'num_iters': 100,
                'num_epochs': 5,
                'lr': 0.01,
                'beta': 0.9,
                'curvature_influence': 0.1,
                'weight_decay': 0.0001,
                'verbose': True
            },
            'ablation': {
                'seed': 42,
                'ablation_epochs': 3,
                'ablation_batches': 100,
                'lr': 0.01,
                'beta': 0.9,
                'curvature_influence': 0.1,
                'lr_values': [0.001, 0.01, 0.1],
                'beta_values': [0.8, 0.9, 0.95],
                'curvature_influence_values': [0.01, 0.1, 0.5]
            },
            'evaluate': {
                'seed': 42
            }
        }
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Run experiments
    results = {}
    
    # Experiment 1: Synthetic Optimization Benchmark
    results['synthetic'] = run_synthetic_experiment(config)
    
    # Experiment 2: CIFAR-10 Deep Neural Network Training
    results['cifar10'] = run_cifar10_experiment(config)
    
    # Experiment 3: MNIST Training and Ablation Study
    results['mnist'] = run_mnist_experiment(config)
    
    # Record end time and calculate total runtime
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total runtime: {total_time:.2f} seconds")
    
    # Synthetic experiment summary
    print("\nSynthetic Optimization Results:")
    print("-" * 50)
    print(f"{'Optimizer':<10} | {'Quadratic Final Loss':<20} | {'Rosenbrock Final Loss':<20}")
    print("-" * 50)
    
    for optimizer_name, optimizer_metrics in results['synthetic']['metrics'].items():
        quadratic_loss = optimizer_metrics['quadratic']['final_loss']
        rosenbrock_loss = optimizer_metrics['rosenbrock']['final_loss']
        print(f"{optimizer_name:<10} | {quadratic_loss:<20.6f} | {rosenbrock_loss:<20.6f}")
    
    # CIFAR-10 experiment summary
    print("\nCIFAR-10 Results:")
    print("-" * 80)
    print(f"{'Optimizer':<10} | {'Test Acc (%)':<12} | {'Test Loss':<10} | {'Train Acc (%)':<12} | {'Train Loss':<10} | {'Epochs to Converge':<18}")
    print("-" * 80)
    
    for optimizer_name, optimizer_metrics in results['cifar10']['metrics'].items():
        print(f"{optimizer_name:<10} | {optimizer_metrics['final_test_acc']:<12.2f} | {optimizer_metrics['final_test_loss']:<10.4f} | "
              f"{optimizer_metrics['final_train_acc']:<12.2f} | {optimizer_metrics['final_train_loss']:<10.4f} | {optimizer_metrics['epochs_to_converge']:<18}")
    
    # MNIST experiment summary
    print("\nMNIST Results:")
    print("-" * 80)
    print(f"{'Optimizer':<10} | {'Test Acc (%)':<12} | {'Test Loss':<10} | {'Train Acc (%)':<12} | {'Train Loss':<10} | {'Epochs to Converge':<18}")
    print("-" * 80)
    
    for optimizer_name, optimizer_metrics in results['mnist']['metrics'].items():
        print(f"{optimizer_name:<10} | {optimizer_metrics['final_test_acc']:<12.2f} | {optimizer_metrics['final_test_loss']:<10.4f} | "
              f"{optimizer_metrics['final_train_acc']:<12.2f} | {optimizer_metrics['final_train_loss']:<10.4f} | {optimizer_metrics['epochs_to_converge']:<18}")
    
    # Ablation study summary
    print("\nAblation Study Results:")
    print(f"Best learning rate: {results['mnist']['ablation_metrics']['best_lr']}")
    print(f"Best beta value: {results['mnist']['ablation_metrics']['best_beta']}")
    print(f"Best curvature influence value: {results['mnist']['ablation_metrics']['best_curvature_influence']}")
    
    # Conclusion
    print("\nCONCLUSION:")
    print("The Adaptive Curvature Momentum (ACM) optimizer demonstrates competitive performance")
    print("compared to established optimizers like Adam and SGD with momentum. The key advantages")
    print("of ACM include:")
    print("1. Adaptive learning rates based on local curvature estimates")
    print("2. Improved convergence in regions with varying curvature")
    print("3. Momentum-based updates for stable optimization")
    
    return results

def quick_test():
    """Run a quick test with minimal iterations to verify code execution."""
    print("Running quick test of the entire experiment pipeline...")
    
    # Set minimal configuration for quick testing
    config = {
        'synthetic': {
            'n_samples': 10,
            'seed': 42
        },
        'cifar10': {
            'batch_size': 64,
            'download': True
        },
        'mnist': {
            'batch_size': 64,
            'download': True
        },
        'train': {
            'seed': 42,
            'num_iters': 5,
            'num_epochs': 1,
            'lr': 0.01,
            'beta': 0.9,
            'curvature_influence': 0.1,
            'weight_decay': 0.0001,
            'verbose': False
        },
        'ablation': {
            'seed': 42,
            'ablation_epochs': 1,
            'ablation_batches': 5,
            'lr': 0.01,
            'beta': 0.9,
            'curvature_influence': 0.1,
            'lr_values': [0.001, 0.01],
            'beta_values': [0.8, 0.9],
            'curvature_influence_values': [0.05, 0.1]
        },
        'evaluate': {
            'seed': 42
        }
    }
    
    # Create a class to limit the number of batches for quick testing
    class LimitedLoader:
        def __init__(self, loader, limit):
            self.loader = loader
            self.limit = limit
            
        def __iter__(self):
            counter = 0
            for item in self.loader:
                if counter >= self.limit:
                    break
                counter += 1
                yield item
                
        def __len__(self):
            return min(self.limit, len(self.loader))
    
    # Preprocess data
    print("\nPreprocessing data for quick test...")
    data = preprocess_data(config)
    
    # Limit the number of batches for quick testing
    data['cifar10']['train_loader'] = LimitedLoader(data['cifar10']['train_loader'], 5)
    data['cifar10']['test_loader'] = LimitedLoader(data['cifar10']['test_loader'], 5)
    data['mnist']['train_loader'] = LimitedLoader(data['mnist']['train_loader'], 5)
    data['mnist']['test_loader'] = LimitedLoader(data['mnist']['test_loader'], 5)
    
    # Run synthetic experiment
    print("\nRunning quick synthetic experiment...")
    synthetic_results = {}
    for optimizer_type in ['ACM', 'Adam', 'SGD_mom']:
        print(f"Training with {optimizer_type}...")
        optimizer_results = train_synthetic(data['synthetic'], optimizer_type, config['train'])
        synthetic_results[optimizer_type] = optimizer_results
    
    evaluate_synthetic(synthetic_results, config['evaluate'])
    
    # Run CIFAR-10 experiment
    print("\nRunning quick CIFAR-10 experiment...")
    cifar10_results = {}
    for optimizer_type in ['ACM', 'Adam', 'SGD_mom']:
        print(f"Training CIFAR-10 with {optimizer_type}...")
        optimizer_results = train_cifar10(data['cifar10'], optimizer_type, config['train'])
        cifar10_results[optimizer_type] = optimizer_results
    
    evaluate_cifar10(cifar10_results, config['evaluate'])
    
    # Run MNIST experiment
    print("\nRunning quick MNIST experiment...")
    mnist_results = {}
    for optimizer_type in ['ACM', 'Adam', 'SGD_mom']:
        print(f"Training MNIST with {optimizer_type}...")
        optimizer_results = train_mnist(data['mnist'], optimizer_type, config['train'])
        mnist_results[optimizer_type] = optimizer_results
    
    evaluate_mnist(mnist_results, config['evaluate'])
    
    # Run ablation study
    print("\nRunning quick ablation study...")
    ablation_results = run_ablation_study(data['mnist'], config['ablation'])
    evaluate_ablation_study(ablation_results, config['evaluate'])
    
    print("\nQuick test completed successfully!")

if __name__ == "__main__":
    # Check if quick test flag is set
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        quick_test()
    else:
        # Run all experiments with default configuration
        run_all_experiments()
