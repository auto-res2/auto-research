"""
Main module for the auto-research project.
Orchestrates the execution of experiments for the Adaptive Curvature Momentum (ACM) optimizer.
"""

import os
import sys
import argparse
import yaml
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.preprocess import set_seed, get_device, load_cifar10, generate_two_moons_data
from src.train import train_cifar10_model, run_synthetic_experiment, train_two_moons_model
from src.evaluate import (
    evaluate_cifar10_model, plot_confusion_matrix, plot_training_history,
    visualize_synthetic_trajectories, plot_two_moons_decision_boundary,
    compare_ablation_results
)
from src.utils.optimizers import ACM

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_experiment(config):
    """
    Set up the experiment environment.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        torch.device: Device to use for the experiment
    """
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Create output directories
    os.makedirs(config.get('log_dir', './logs'), exist_ok=True)
    os.makedirs(config.get('model_dir', './models'), exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Print CUDA information if available
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

def run_cifar10_experiment(config):
    """
    Run the CIFAR-10 experiment with ResNet-18.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("Experiment 1: CIFAR-10 with ResNet-18")
    print("="*80)
    
    # Load configuration
    cifar_config = config.get('cifar10', {})
    optimizer_config = config.get('optimizer', {})
    
    if not cifar_config.get('enabled', True):
        print("Experiment disabled in configuration. Skipping.")
        return
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader, classes = load_cifar10(
        batch_size=cifar_config.get('batch_size', 128),
        num_workers=cifar_config.get('num_workers', 2)
    )
    print(f"Dataset loaded. Training set: {len(trainloader.dataset)} images, "
          f"Test set: {len(testloader.dataset)} images")
    
    # Train with ACM
    print("\nTraining with ACM optimizer...")
    acm_model, acm_losses, acm_accuracies = train_cifar10_model(
        trainloader=trainloader,
        testloader=testloader,
        optimizer_name='acm',
        num_epochs=cifar_config.get('num_epochs', 3),
        lr=optimizer_config.get('learning_rate', 0.001),
        beta=optimizer_config.get('beta', 0.9),
        curvature_scale=optimizer_config.get('curvature_scale', 1.0),
        log_dir=config.get('log_dir', './logs'),
        model_dir=config.get('model_dir', './models')
    )
    
    # Evaluate ACM model
    print("\nEvaluating ACM model...")
    acm_metrics = evaluate_cifar10_model(acm_model, testloader)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        acm_metrics['confusion_matrix'],
        classes,
        save_path=f"{config.get('log_dir', './logs')}/cifar10_acm_confusion_matrix.png"
    )
    
    # Plot training history
    plot_training_history(
        acm_losses,
        acm_accuracies,
        'ACM',
        save_path=f"{config.get('log_dir', './logs')}/cifar10_acm_history.png"
    )
    
    # Compare with Adam if enabled
    if cifar_config.get('compare_with_adam', True):
        print("\nTraining with Adam optimizer for comparison...")
        adam_model, adam_losses, adam_accuracies = train_cifar10_model(
            trainloader=trainloader,
            testloader=testloader,
            optimizer_name='adam',
            num_epochs=cifar_config.get('num_epochs', 3),
            lr=optimizer_config.get('learning_rate', 0.001),
            log_dir=config.get('log_dir', './logs'),
            model_dir=config.get('model_dir', './models')
        )
        
        # Evaluate Adam model
        print("\nEvaluating Adam model...")
        adam_metrics = evaluate_cifar10_model(adam_model, testloader)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            adam_metrics['confusion_matrix'],
            classes,
            save_path=f"{config.get('log_dir', './logs')}/cifar10_adam_confusion_matrix.png"
        )
        
        # Plot training history
        plot_training_history(
            adam_losses,
            adam_accuracies,
            'Adam',
            save_path=f"{config.get('log_dir', './logs')}/cifar10_adam_history.png"
        )
        
        # Print comparison
        print("\nComparison of ACM vs Adam:")
        print(f"ACM Final Accuracy: {acm_accuracies[-1]:.2f}%")
        print(f"Adam Final Accuracy: {adam_accuracies[-1]:.2f}%")
        print(f"Difference: {acm_accuracies[-1] - adam_accuracies[-1]:.2f}%")
    
    print("\nCIFAR-10 experiment completed.")

def run_synthetic_experiment(config):
    """
    Run the synthetic function optimization experiment.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("Experiment 2: Synthetic Function Optimization")
    print("="*80)
    
    # Load configuration
    synthetic_config = config.get('synthetic', {})
    optimizer_config = config.get('optimizer', {})
    
    if not synthetic_config.get('enabled', True):
        print("Experiment disabled in configuration. Skipping.")
        return
    
    # Get parameters
    num_steps = synthetic_config.get('num_steps', 50)
    starting_point = synthetic_config.get('starting_point', [4.0, 4.0])
    function_params = synthetic_config.get('function_params', {
        'a': 1.0,
        'b': 2.0,
        'c': -1.0
    })
    
    # Run ACM optimizer
    print("\nRunning synthetic optimization with ACM...")
    trajectory_acm = run_synthetic_experiment(
        optimizer_class=ACM,
        optimizer_kwargs={
            'lr': optimizer_config.get('learning_rate', 0.1),
            'beta': optimizer_config.get('beta', 0.9),
            'curvature_scale': optimizer_config.get('curvature_scale', 1.0)
        },
        num_steps=num_steps
    )
    
    # Compare with Adam if enabled
    if synthetic_config.get('compare_with_adam', True):
        print("\nRunning synthetic optimization with Adam for comparison...")
        trajectory_adam = run_synthetic_experiment(
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={'lr': optimizer_config.get('learning_rate', 0.1)},
            num_steps=num_steps
        )
        
        # Visualize trajectories
        print("\nVisualizing optimizer trajectories...")
        visualize_synthetic_trajectories(
            trajectory_acm,
            trajectory_adam,
            save_path=f"{config.get('log_dir', './logs')}/synthetic_trajectories.png"
        )
        
        # Print comparison
        print("\nComparison of final positions:")
        print(f"ACM final position: [{trajectory_acm[-1, 0]:.4f}, {trajectory_acm[-1, 1]:.4f}]")
        print(f"Adam final position: [{trajectory_adam[-1, 0]:.4f}, {trajectory_adam[-1, 1]:.4f}]")
    else:
        # Visualize only ACM trajectory
        print("\nVisualizing ACM optimizer trajectory...")
        # Create a dummy trajectory for Adam (just for visualization)
        trajectory_adam = np.zeros_like(trajectory_acm)
        visualize_synthetic_trajectories(
            trajectory_acm,
            trajectory_adam,
            save_path=f"{config.get('log_dir', './logs')}/synthetic_trajectories.png"
        )
    
    print("\nSynthetic function optimization experiment completed.")

def run_two_moons_experiment(config):
    """
    Run the two-moons classification experiment with hyperparameter ablation.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("Experiment 3: Two-Moons Classification with Hyperparameter Ablation")
    print("="*80)
    
    # Load configuration
    two_moons_config = config.get('two_moons', {})
    optimizer_config = config.get('optimizer', {})
    
    if not two_moons_config.get('enabled', True):
        print("Experiment disabled in configuration. Skipping.")
        return
    
    # Generate data
    print("\nGenerating two-moons dataset...")
    X_train, y_train, X_val, y_val = generate_two_moons_data(
        n_samples=two_moons_config.get('n_samples', 1000),
        noise=two_moons_config.get('noise', 0.2),
        test_size=two_moons_config.get('test_size', 0.2),
        seed=config.get('seed', 42)
    )
    print(f"Dataset generated. Training set: {X_train.shape[0]} samples, "
          f"Validation set: {X_val.shape[0]} samples")
    
    # Run ablation study
    print("\nRunning hyperparameter ablation study...")
    results = {}
    
    # Get hyperparameter values to test
    beta_values = two_moons_config.get('beta_values', [0.8, 0.9, 0.99])
    cs_values = two_moons_config.get('curvature_scale_values', [0.5, 1.0, 2.0])
    
    for beta in beta_values:
        for cs in cs_values:
            key = f"beta_{beta}_cs_{cs}"
            print(f"\nTraining with {key}...")
            
            model, losses, accuracy = train_two_moons_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                optimizer_name='acm',
                num_epochs=two_moons_config.get('num_epochs', 50),
                lr=two_moons_config.get('learning_rate', 0.05),
                beta=beta,
                curvature_scale=cs,
                hidden_dim=two_moons_config.get('hidden_dim', 10),
                log_dir=config.get('log_dir', './logs'),
                model_dir=config.get('model_dir', './models')
            )
            
            results[key] = {
                'losses': losses,
                'accuracy': accuracy,
                'model': model
            }
            
            # Plot decision boundary
            if two_moons_config.get('save_visualization', True):
                plot_two_moons_decision_boundary(
                    model,
                    X_val,
                    y_val,
                    save_path=f"{config.get('log_dir', './logs')}/two_moons_{key}_boundary.png"
                )
    
    # Compare results
    print("\nComparing ablation study results...")
    compare_ablation_results(
        results,
        save_path=f"{config.get('log_dir', './logs')}/two_moons_ablation_comparison.png"
    )
    
    # Find best configuration
    best_key = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_beta = float(best_key.split('_')[1])
    best_cs = float(best_key.split('_')[3])
    best_accuracy = results[best_key]['accuracy']
    
    print("\nBest hyperparameter configuration:")
    print(f"Beta: {best_beta}")
    print(f"Curvature Scale: {best_cs}")
    print(f"Validation Accuracy: {best_accuracy:.4f}")
    
    # Compare with Adam if enabled
    if two_moons_config.get('compare_with_adam', True):
        print("\nTraining with Adam optimizer for comparison...")
        adam_model, adam_losses, adam_accuracy = train_two_moons_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            optimizer_name='adam',
            num_epochs=two_moons_config.get('num_epochs', 50),
            lr=two_moons_config.get('learning_rate', 0.05),
            log_dir=config.get('log_dir', './logs'),
            model_dir=config.get('model_dir', './models')
        )
        
        # Plot decision boundary
        if two_moons_config.get('save_visualization', True):
            plot_two_moons_decision_boundary(
                adam_model,
                X_val,
                y_val,
                save_path=f"{config.get('log_dir', './logs')}/two_moons_adam_boundary.png"
            )
        
        # Print comparison
        print("\nComparison of best ACM vs Adam:")
        print(f"Best ACM Validation Accuracy: {best_accuracy:.4f}")
        print(f"Adam Validation Accuracy: {adam_accuracy:.4f}")
        print(f"Difference: {best_accuracy - adam_accuracy:.4f}")
    
    print("\nTwo-moons experiment completed.")

def main():
    """
    Main function to run the ACM optimizer experiments.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run ACM optimizer experiments')
    parser.add_argument('--config', type=str, default='./config/acm_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'cifar10', 'synthetic', 'two_moons'],
                        help='Experiment to run')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Set up experiment
    device = setup_experiment(config)
    
    # Print experiment information
    print("\n" + "="*80)
    print("Adaptive Curvature Momentum (ACM) Optimizer Experiments")
    print("="*80)
    print(f"Configuration file: {args.config}")
    print(f"Experiment(s) to run: {args.experiment}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Run experiments
    start_time = time.time()
    
    if args.experiment in ['all', 'cifar10']:
        run_cifar10_experiment(config)
    
    if args.experiment in ['all', 'synthetic']:
        run_synthetic_experiment(config)
    
    if args.experiment in ['all', 'two_moons']:
        run_two_moons_experiment(config)
    
    # Print total execution time
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"All experiments completed in {total_time:.2f} seconds")
    print("="*80)

if __name__ == '__main__':
    main()
