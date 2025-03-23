"""Main script for running the ACM optimizer experiment."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from .preprocess import load_cifar10, load_mnist
from .train import train_model, run_synthetic_experiment
from .evaluate import evaluate_model, plot_training_curves, plot_synthetic_trajectories
from .utils.models import SimpleCNNCIFAR10, SimpleCNNMNIST
from .utils.optimizers import ACMOptimizer
from config.experiment_config import EXPERIMENT_CONFIG, MODEL_CONFIG


def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)


def run_synthetic_experiments():
    """Run experiments on synthetic optimization problems."""
    print("\n" + "="*80)
    print("SYNTHETIC OPTIMIZATION EXPERIMENTS")
    print("="*80)
    
    num_iters = EXPERIMENT_CONFIG['synthetic']['num_iters']
    optimizers = ['acm', 'adam', 'sgd_mom']
    
    # Run experiments for each optimizer
    results = {}
    for opt_name in optimizers:
        results[opt_name] = run_synthetic_experiment(opt_name, num_iters)
    
    # Plot and compare results
    plot_synthetic_trajectories(
        results, 
        "Synthetic Quadratic Optimization", 
        save_path="logs/synthetic_experiment.png"
    )
    
    return results


def run_cifar10_experiments():
    """Run experiments on CIFAR-10 dataset."""
    print("\n" + "="*80)
    print("CIFAR-10 EXPERIMENTS")
    print("="*80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    batch_size = EXPERIMENT_CONFIG['cifar10']['batch_size']
    train_loader, test_loader = load_cifar10(batch_size=batch_size)
    print(f"Data loaded: {len(train_loader.dataset)} training samples, "
          f"{len(test_loader.dataset)} test samples")
    
    # Setup optimizers to compare
    optimizers = ['acm', 'adam', 'sgd_mom']
    results = {}
    
    # Run training for each optimizer
    for opt_name in optimizers:
        print(f"\nTraining with {opt_name} optimizer")
        model = SimpleCNNCIFAR10(MODEL_CONFIG['cnn_cifar10']).to(device)
        
        # Train the model
        train_results = train_model(
            model=model,
            train_loader=train_loader,
            optimizer_name=opt_name,
            device=device,
            epochs=EXPERIMENT_CONFIG['cifar10']['num_epochs'],
            log_interval=EXPERIMENT_CONFIG['cifar10']['log_interval']
        )
        
        # Evaluate the model
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)
        
        # Save model
        torch.save(model.state_dict(), f"models/cifar10_{opt_name}.pt")
        
        # Store results
        results[opt_name] = {
            'train': train_results,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    # Plot and compare results
    plot_training_curves(
        {k: v['train'] for k, v in results.items()},
        "CIFAR-10 Training",
        save_path="logs/cifar10_training.png"
    )
    
    return results


def run_mnist_experiments():
    """Run experiments on MNIST dataset."""
    print("\n" + "="*80)
    print("MNIST EXPERIMENTS")
    print("="*80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    batch_size = EXPERIMENT_CONFIG['mnist']['batch_size']
    train_loader, test_loader = load_mnist(batch_size=batch_size)
    print(f"Data loaded: {len(train_loader.dataset)} training samples, "
          f"{len(test_loader.dataset)} test samples")
    
    # Setup optimizers to compare
    optimizers = ['acm', 'adam', 'sgd_mom']
    results = {}
    
    # Run training for each optimizer
    for opt_name in optimizers:
        print(f"\nTraining with {opt_name} optimizer")
        model = SimpleCNNMNIST(MODEL_CONFIG['cnn_mnist']).to(device)
        
        # Train the model
        train_results = train_model(
            model=model,
            train_loader=train_loader,
            optimizer_name=opt_name,
            device=device,
            epochs=EXPERIMENT_CONFIG['mnist']['num_epochs'],
            log_interval=EXPERIMENT_CONFIG['mnist']['log_interval']
        )
        
        # Evaluate the model
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)
        
        # Save model
        torch.save(model.state_dict(), f"models/mnist_{opt_name}.pt")
        
        # Store results
        results[opt_name] = {
            'train': train_results,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    # Plot and compare results
    plot_training_curves(
        {k: v['train'] for k, v in results.items()},
        "MNIST Training",
        save_path="logs/mnist_training.png"
    )
    
    return results


def quick_test():
    """Run a quick test to verify the implementation."""
    print("\n" + "="*80)
    print("QUICK TEST")
    print("="*80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run a synthetic experiment with fewer iterations
    print("Running synthetic experiment...")
    synthetic_results = run_synthetic_experiment('acm', num_iters=10)
    
    print("Quick test completed successfully!")
    return True


def main():
    """Main function to run the experiments."""
    print("="*80)
    print("ADAPTIVE CURVATURE MOMENTUM (ACM) OPTIMIZER EXPERIMENTS")
    print("="*80)
    
    # Setup directories
    setup_directories()
    
    # Check if quick test flag is set
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--quick-test':
        return quick_test()
    
    # Run synthetic experiments
    synthetic_results = run_synthetic_experiments()
    
    # Run CIFAR-10 experiments
    cifar10_results = run_cifar10_experiments()
    
    # Run MNIST experiments
    mnist_results = run_mnist_experiments()
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Print summary of results
    print("\nSynthetic Experiment:")
    for opt_name, results in synthetic_results.items():
        print(f"  {opt_name}: Final loss = {results['losses'][-1]:.6f}")
    
    print("\nCIFAR-10 Experiment:")
    for opt_name, results in cifar10_results.items():
        print(f"  {opt_name}: Test accuracy = {results['test_accuracy']:.2f}%")
    
    print("\nMNIST Experiment:")
    for opt_name, results in mnist_results.items():
        print(f"  {opt_name}: Test accuracy = {results['test_accuracy']:.2f}%")
    
    print("\nExperiments completed successfully!")
    print("Results and plots saved in the logs directory.")
    
    return 0


if __name__ == "__main__":
    main()
