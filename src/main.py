"""
Main script for running ACM optimizer experiments.

This script implements the entire process from data preprocessing to model training
and evaluation, comparing the Adaptive Curvature Momentum (ACM) optimizer with
other standard optimizers.
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.preprocess import prepare_data
from src.train import run_experiment
from src.evaluate import evaluate_experiment


def setup_experiment_config(experiment_type):
    """
    Set up configuration for the experiment.
    
    Args:
        experiment_type (str): Type of experiment to run
        
    Returns:
        config: Configuration dictionary
    """
    # Base configuration
    config = {
        'seed': 42,
        'data_dir': './data',
        'save_dir': './models',
        'log_dir': './logs',
        'optimizers': ['sgd', 'adam', 'adabelief', 'acm']
    }
    
    # Create directories
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Experiment-specific configuration
    if experiment_type == 'experiment_1':
        # Experiment 1: Convergence and Generalization on CIFAR-10 with ResNet-18
        config.update({
            'experiment_type': 'cifar10',
            'model_name': 'resnet18',
            'num_classes': 10,
            'batch_size': 128,
            'num_workers': 4,
            'num_epochs': 5,  # Reduced for testing, use 50 for full experiment
            'lr': 0.001,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'curvature_coef': 0.1
        })
    
    elif experiment_type == 'experiment_2':
        # Experiment 2: Curvature Sensitivity Analysis on a Quadratic Function
        config.update({
            'experiment_type': 'quadratic',
            'model_name': 'quadratic',
            'dimension': 10,
            'n_samples': 1000,
            'curvature': 5.0,
            'num_iterations': 20,  # Reduced for testing, use 50 for full experiment
            'lr': 0.1,
            'weight_decay': 0,
            'curvature_coef': 0.1
        })
    
    elif experiment_type == 'experiment_3':
        # Experiment 3: Hyperparameter Robustness with a Simple CNN on CIFAR-10
        config.update({
            'experiment_type': 'hyperparameter_search',
            'model_name': 'simplecnn',
            'num_classes': 10,
            'batch_size': 128,
            'num_workers': 4,
            'num_epochs': 3,  # Reduced for testing, use 20 for full experiment
            'weight_decay': 5e-4
        })
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Save configuration to file
    os.makedirs('./config', exist_ok=True)
    with open(f'./config/{experiment_type}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return config


def experiment_1(test_mode=False):
    """
    Experiment 1: Convergence and Generalization on CIFAR-10 with ResNet-18.
    
    Args:
        test_mode (bool): Whether to run in test mode with reduced epochs
    """
    print("\n" + "="*80)
    print("Experiment 1: Convergence and Generalization on CIFAR-10 with ResNet-18")
    print("="*80)
    
    # Set up configuration
    config = setup_experiment_config('experiment_1')
    
    if test_mode:
        config['num_epochs'] = 1
        config['batch_size'] = 64
    
    # Prepare data
    data = prepare_data(config)
    
    # Run experiment
    results = run_experiment(config)
    
    # Evaluate experiment
    eval_results = evaluate_experiment(config, results, data)
    
    print("\nExperiment 1 completed successfully!")


def experiment_2(test_mode=False):
    """
    Experiment 2: Curvature Sensitivity Analysis on a Quadratic Function.
    
    Args:
        test_mode (bool): Whether to run in test mode with reduced iterations
    """
    print("\n" + "="*80)
    print("Experiment 2: Curvature Sensitivity Analysis on a Quadratic Function")
    print("="*80)
    
    # Set up configuration
    config = setup_experiment_config('experiment_2')
    
    if test_mode:
        config['num_iterations'] = 5
    
    # Prepare data
    data = prepare_data(config)
    
    # Run experiment
    results = run_experiment(config)
    
    # Evaluate experiment
    eval_results = evaluate_experiment(config, results, data)
    
    print("\nExperiment 2 completed successfully!")


def experiment_3(test_mode=False):
    """
    Experiment 3: Hyperparameter Robustness with a Simple CNN on CIFAR-10.
    
    Args:
        test_mode (bool): Whether to run in test mode with reduced epochs
    """
    print("\n" + "="*80)
    print("Experiment 3: Hyperparameter Robustness with a Simple CNN on CIFAR-10")
    print("="*80)
    
    # Set up configuration
    config = setup_experiment_config('experiment_3')
    
    if test_mode:
        config['num_epochs'] = 1
        config['batch_size'] = 64
    
    # Define hyperparameter grid
    lr_list = [1e-4, 1e-3]
    curvature_coef_list = [0.01, 0.1]
    
    # Prepare data
    base_data = prepare_data(config)
    
    # Run hyperparameter search
    results = {}
    
    for lr in lr_list:
        for curvature_coef in curvature_coef_list:
            print(f"\nTraining with lr={lr} and curvature_coef={curvature_coef}")
            
            # Update configuration
            config['lr'] = lr
            config['curvature_coef'] = curvature_coef
            config['optimizer'] = 'acm'
            
            # Run experiment
            experiment_results = run_experiment(config)
            
            # Store best validation accuracy
            val_acc = max(experiment_results['acm']['val_acc'])
            results[(lr, curvature_coef)] = val_acc
    
    # Evaluate experiment
    eval_results = evaluate_experiment(config, results, base_data)
    
    print("\nExperiment 3 completed successfully!")


def test_experiments():
    """
    Run a quick test version of each experiment to check that the code executes correctly.
    """
    print("\n" + "="*80)
    print("Running Test Experiments")
    print("="*80)
    
    # Run experiments in test mode
    experiment_1(test_mode=True)
    experiment_2(test_mode=True)
    experiment_3(test_mode=True)
    
    print("\nAll test experiments completed successfully!")


def main():
    """
    Main function to run experiments.
    """
    parser = argparse.ArgumentParser(description='Run ACM optimizer experiments')
    parser.add_argument('--experiment', type=str, default='test',
                        choices=['1', '2', '3', 'all', 'test'],
                        help='Experiment to run (1, 2, 3, all, or test)')
    args = parser.parse_args()
    
    # Print system information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run selected experiment
    if args.experiment == '1':
        experiment_1()
    elif args.experiment == '2':
        experiment_2()
    elif args.experiment == '3':
        experiment_3()
    elif args.experiment == 'all':
        experiment_1()
        experiment_2()
        experiment_3()
    else:  # test
        test_experiments()


if __name__ == '__main__':
    main()
