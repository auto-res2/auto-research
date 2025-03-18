import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
from preprocess import get_cifar10_data, get_agnews_data, create_rosenbrock_function
from train import train_cifar10, optimize_rosenbrock, train_text_classifier
from evaluate import (
    evaluate_cifar10_model, 
    plot_cifar10_results, 
    plot_rosenbrock_results, 
    plot_text_classification_results,
    plot_hyperparameter_sensitivity,
    save_experiment_results
)

def run_experiment_1(device=None, num_epochs=5):
    """
    Experiment 1: CIFAR-10 training with ResNet-18
    
    Args:
        device: Device to use for training (None for auto-detection)
        num_epochs: Number of training epochs
        
    Returns:
        dict: Dictionary containing experiment results
    """
    print("\n" + "="*80)
    print("Experiment 1: CIFAR-10 Classification with ResNet-18")
    print("="*80)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_data(batch_size=128, num_workers=2)
    print("Dataset loaded successfully.")
    
    # Train models with different optimizers
    optimizers = ['acm', 'adam', 'sgd']
    histories = {}
    
    for optimizer_name in optimizers:
        print(f"\nTraining with {optimizer_name.upper()} optimizer:")
        
        # Set learning rate based on optimizer
        lr = 0.1 if optimizer_name == 'sgd' else 0.001
        
        # Train model
        history = train_cifar10(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer_name=optimizer_name,
            num_epochs=num_epochs,
            lr=lr,
            beta=0.9,
            save_model=True,
            device=device
        )
        
        histories[optimizer_name] = history
    
    # Plot results
    print("\nPlotting results...")
    plot_cifar10_results(histories, save_path='logs/cifar10_results.png')
    
    # Save results
    results = {
        'experiment': 'cifar10_resnet18',
        'histories': histories
    }
    save_experiment_results(results, filename='logs/cifar10_results.json')
    
    print("\nExperiment 1 completed successfully.")
    return results

def run_experiment_2(num_iters=500):
    """
    Experiment 2: Rosenbrock Function Optimization
    
    Args:
        num_iters: Number of optimization iterations
        
    Returns:
        dict: Dictionary containing experiment results
    """
    print("\n" + "="*80)
    print("Experiment 2: Rosenbrock Function Optimization")
    print("="*80)
    
    # Optimize with different optimizers
    optimizers = ['acm', 'adam', 'sgd']
    trajectories = {}
    loss_values = {}
    
    for optimizer_name in optimizers:
        print(f"\nOptimizing with {optimizer_name.upper()} optimizer:")
        
        # Set learning rate based on optimizer
        lr = 0.01 if optimizer_name == 'sgd' else 0.001
        
        # Optimize function
        trajectory, losses = optimize_rosenbrock(
            optimizer_name=optimizer_name,
            num_iters=num_iters,
            lr=lr,
            beta=0.9
        )
        
        trajectories[optimizer_name] = trajectory
        loss_values[optimizer_name] = losses
    
    # Plot results
    print("\nPlotting results...")
    plot_rosenbrock_results(
        trajectories=trajectories,
        loss_values=loss_values,
        save_path='logs/rosenbrock_results.png'
    )
    
    # Save results
    results = {
        'experiment': 'rosenbrock_optimization',
        'trajectories': trajectories,
        'loss_values': loss_values
    }
    save_experiment_results(results, filename='logs/rosenbrock_results.json')
    
    print("\nExperiment 2 completed successfully.")
    return results

def run_experiment_3(device=None, num_epochs=5, grid_size=3):
    """
    Experiment 3: Hyperparameter Sensitivity Analysis on Text Classification
    
    Args:
        device: Device to use for training (None for auto-detection)
        num_epochs: Number of training epochs
        grid_size: Size of the hyperparameter grid (1, 2, or 3)
        
    Returns:
        dict: Dictionary containing experiment results
    """
    print("\n" + "="*80)
    print("Experiment 3: Hyperparameter Sensitivity Analysis on Text Classification")
    print("="*80)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading AG_NEWS dataset...")
    train_loader, test_loader, vocab = get_agnews_data(batch_size=64)
    print("Dataset loaded successfully.")
    
    # Define hyperparameter grid
    if grid_size == 1:
        lr_values = [0.001]
        beta_values = [0.9]
    elif grid_size == 2:
        lr_values = [0.0001, 0.001]
        beta_values = [0.5, 0.9]
    else:  # grid_size == 3
        lr_values = [0.0001, 0.001, 0.01]
        beta_values = [0.5, 0.9, 0.99]
    
    # Train models with different hyperparameters
    param_histories = {}
    
    # ACM optimizer with different hyperparameters
    for lr in lr_values:
        for beta in beta_values:
            print(f"\nTraining with ACM optimizer (lr={lr}, beta={beta}):")
            
            # Train model
            history = train_text_classifier(
                train_loader=train_loader,
                test_loader=test_loader,
                vocab=vocab,
                optimizer_name='acm',
                num_epochs=num_epochs,
                lr=lr,
                beta=beta,
                device=device
            )
            
            param_histories[('acm', 'lr', lr, 'beta', beta)] = history
    
    # Adam optimizer with different learning rates
    for lr in lr_values:
        print(f"\nTraining with Adam optimizer (lr={lr}):")
        
        # Train model
        history = train_text_classifier(
            train_loader=train_loader,
            test_loader=test_loader,
            vocab=vocab,
            optimizer_name='adam',
            num_epochs=num_epochs,
            lr=lr,
            beta=0.9,  # Not used for Adam
            device=device
        )
        
        param_histories[('adam', 'lr', lr)] = history
    
    # SGD optimizer with different learning rates
    for lr in lr_values:
        print(f"\nTraining with SGD optimizer (lr={lr}):")
        
        # Train model
        history = train_text_classifier(
            train_loader=train_loader,
            test_loader=test_loader,
            vocab=vocab,
            optimizer_name='sgd',
            num_epochs=num_epochs,
            lr=lr,
            beta=0.9,  # Not used for SGD
            device=device
        )
        
        param_histories[('sgd', 'lr', lr)] = history
    
    # Plot results
    print("\nPlotting results...")
    
    # Group histories by optimizer for comparison
    acm_histories = {}
    adam_histories = {}
    sgd_histories = {}
    
    for params, history in param_histories.items():
        if params[0] == 'acm':
            acm_histories[f"lr={params[2]}, beta={params[4]}"] = history
        elif params[0] == 'adam':
            adam_histories[f"lr={params[2]}"] = history
        elif params[0] == 'sgd':
            sgd_histories[f"lr={params[2]}"] = history
    
    # Plot optimizer comparisons
    best_acm_history = None
    best_acm_params = None
    best_acm_acc = 0
    
    for params, history in acm_histories.items():
        if history['test_acc'][-1] > best_acm_acc:
            best_acm_acc = history['test_acc'][-1]
            best_acm_history = history
            best_acm_params = params
    
    best_adam_history = None
    best_adam_params = None
    best_adam_acc = 0
    
    for params, history in adam_histories.items():
        if history['test_acc'][-1] > best_adam_acc:
            best_adam_acc = history['test_acc'][-1]
            best_adam_history = history
            best_adam_params = params
    
    best_sgd_history = None
    best_sgd_params = None
    best_sgd_acc = 0
    
    for params, history in sgd_histories.items():
        if history['test_acc'][-1] > best_sgd_acc:
            best_sgd_acc = history['test_acc'][-1]
            best_sgd_history = history
            best_sgd_params = params
    
    # Compare best configurations
    best_histories = {
        f"ACM ({best_acm_params})": best_acm_history,
        f"Adam ({best_adam_params})": best_adam_history,
        f"SGD ({best_sgd_params})": best_sgd_history
    }
    
    plot_text_classification_results(
        histories=best_histories,
        save_path='logs/text_classification_results.png'
    )
    
    # Plot hyperparameter sensitivity
    plot_hyperparameter_sensitivity(
        param_histories=param_histories,
        save_path='logs/hyperparameter_sensitivity.png'
    )
    
    # Save results
    results = {
        'experiment': 'text_classification_hyperparameter_sensitivity',
        'param_histories': param_histories,
        'best_histories': best_histories
    }
    save_experiment_results(results, filename='logs/text_classification_results.json')
    
    print("\nExperiment 3 completed successfully.")
    return results

def run_quick_test():
    """
    Run a quick test of all experiments with minimal iterations
    to verify that the code works correctly.
    """
    print("\n" + "="*80)
    print("Running Quick Test of All Experiments")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiment 1 with 1 epoch
    print("\nRunning quick test of Experiment 1...")
    run_experiment_1(device=device, num_epochs=1)
    
    # Run experiment 2 with 50 iterations
    print("\nRunning quick test of Experiment 2...")
    run_experiment_2(num_iters=50)
    
    # Run experiment 3 with 1 epoch and grid size 1
    print("\nRunning quick test of Experiment 3...")
    run_experiment_3(device=device, num_epochs=1, grid_size=1)
    
    print("\nQuick test completed successfully.")

def main():
    """
    Main function to run all experiments
    """
    # Create directories for outputs
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run ACM optimizer experiments')
    parser.add_argument('--test', action='store_true', help='Run quick test of all experiments')
    parser.add_argument('--exp1', action='store_true', help='Run experiment 1: CIFAR-10 with ResNet-18')
    parser.add_argument('--exp2', action='store_true', help='Run experiment 2: Rosenbrock optimization')
    parser.add_argument('--exp3', action='store_true', help='Run experiment 3: Hyperparameter sensitivity')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--iters', type=int, default=500, help='Number of iterations for optimization')
    parser.add_argument('--grid-size', type=int, default=3, help='Size of hyperparameter grid (1, 2, or 3)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiments based on arguments
    if args.test:
        run_quick_test()
    elif args.all or (not args.exp1 and not args.exp2 and not args.exp3):
        # Run all experiments if --all is specified or no specific experiment is selected
        run_experiment_1(device=device, num_epochs=args.epochs)
        run_experiment_2(num_iters=args.iters)
        run_experiment_3(device=device, num_epochs=args.epochs, grid_size=args.grid_size)
    else:
        # Run specific experiments
        if args.exp1:
            run_experiment_1(device=device, num_epochs=args.epochs)
        if args.exp2:
            run_experiment_2(num_iters=args.iters)
        if args.exp3:
            run_experiment_3(device=device, num_epochs=args.epochs, grid_size=args.grid_size)
    
    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(f"All experiments completed in {duration:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()
