import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_cifar10, load_mnist
from src.train import (
    train_synthetic, train_rosenbrock, train_cifar10, 
    train_mnist_ablation
)
from src.evaluate import (
    evaluate_synthetic_results, evaluate_cifar10, 
    evaluate_mnist_ablation
)
from src.utils.optimizers import ACMOptimizer
from config.experiment_config import (
    SYNTHETIC_CONFIG, CIFAR10_CONFIG, MNIST_CONFIG, 
    QUICK_TEST_CONFIG
)

def setup_directories():
    """
    Create necessary directories if they don't exist
    """
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

def run_synthetic_experiments(quick_test=False):
    """
    Run synthetic optimization experiments
    """
    print("\n" + "="*80)
    print("SYNTHETIC OPTIMIZATION EXPERIMENTS")
    print("="*80)
    
    # Quadratic function experiment
    quadratic_results = {}
    for optimizer_name in ["ACM", "Adam", "SGD_mom"]:
        losses, final_point = train_synthetic(
            optimizer_name, 
            num_iters=SYNTHETIC_CONFIG['quadratic']['num_iters'] if not quick_test else QUICK_TEST_CONFIG['synthetic_iters'],
            quick_test=quick_test
        )
        quadratic_results[optimizer_name] = losses
        print(f"Final point for {optimizer_name}: {final_point}")
    
    evaluate_synthetic_results(quadratic_results, "Quadratic Function")
    
    # Rosenbrock function experiment
    rosenbrock_results = {}
    for optimizer_name in ["ACM", "Adam", "SGD_mom"]:
        losses, final_point = train_rosenbrock(
            optimizer_name, 
            num_iters=SYNTHETIC_CONFIG['rosenbrock']['num_iters'] if not quick_test else QUICK_TEST_CONFIG['synthetic_iters'],
            quick_test=quick_test
        )
        rosenbrock_results[optimizer_name] = losses
        print(f"Final point for {optimizer_name}: {final_point}")
    
    evaluate_synthetic_results(rosenbrock_results, "Rosenbrock Function")
    
    return quadratic_results, rosenbrock_results

def run_cifar10_experiments(quick_test=False):
    """
    Run CIFAR-10 experiments
    """
    print("\n" + "="*80)
    print("CIFAR-10 EXPERIMENTS")
    print("="*80)
    
    # Load CIFAR-10 dataset
    trainloader, testloader = load_cifar10(
        batch_size=CIFAR10_CONFIG['batch_size'],
        test_batch_size=CIFAR10_CONFIG['test_batch_size']
    )
    
    # Train models with different optimizers
    results = {}
    for optimizer_name in ["ACM", "Adam", "SGD_mom"]:
        train_losses, train_accs, test_accs = train_cifar10(
            optimizer_name, 
            trainloader, 
            testloader, 
            epochs=CIFAR10_CONFIG['epochs'] if not quick_test else QUICK_TEST_CONFIG['cifar10_epochs'],
            quick_test=quick_test
        )
        results[optimizer_name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
        }
    
    # Plot training loss curves
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['train_losses'], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("CIFAR-10 Training Loss")
    plt.legend()
    plt.savefig('./logs/cifar10_train_loss.png')
    plt.close()
    
    # Plot test accuracy curves
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['test_accs'], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("CIFAR-10 Test Accuracy")
    plt.legend()
    plt.savefig('./logs/cifar10_test_acc.png')
    plt.close()
    
    # Evaluate models
    model_paths = [f'./models/cifar10_{name}.pth' for name in ["ACM", "Adam", "SGD_mom"]]
    evaluation_results = evaluate_cifar10(model_paths, testloader)
    
    return results, evaluation_results

def run_mnist_ablation_study(quick_test=False):
    """
    Run MNIST ablation study for ACM optimizer
    """
    print("\n" + "="*80)
    print("MNIST ABLATION STUDY")
    print("="*80)
    
    # Load MNIST dataset
    trainloader, testloader = load_mnist(
        batch_size=MNIST_CONFIG['batch_size'],
        test_batch_size=MNIST_CONFIG['test_batch_size']
    )
    
    # Run ablation study
    results = train_mnist_ablation(
        trainloader, 
        testloader, 
        lr_values=MNIST_CONFIG['lr_values'],
        beta_values=MNIST_CONFIG['beta_values'],
        curvature_values=MNIST_CONFIG['curvature_values'],
        epochs=MNIST_CONFIG['epochs'] if not quick_test else QUICK_TEST_CONFIG['mnist_epochs'],
        quick_test=quick_test
    )
    
    # Evaluate results
    best_config = evaluate_mnist_ablation(results)
    
    return results, best_config

def main():
    """
    Main function to run all experiments
    """
    print("="*80)
    print("ADAPTIVE CURVATURE MOMENTUM (ACM) OPTIMIZER EXPERIMENTS")
    print("="*80)
    
    # Create necessary directories
    setup_directories()
    
    # Check if quick test mode is enabled
    quick_test = QUICK_TEST_CONFIG['enabled']
    if quick_test:
        print("\nRunning in QUICK TEST mode (reduced iterations/epochs for verification)")
    else:
        print("\nRunning FULL experiments")
    
    # Run synthetic optimization experiments
    quadratic_results, rosenbrock_results = run_synthetic_experiments(quick_test)
    
    # Run CIFAR-10 experiments
    cifar10_results, cifar10_eval = run_cifar10_experiments(quick_test)
    
    # Run MNIST ablation study
    mnist_results, best_mnist_config = run_mnist_ablation_study(quick_test)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print("\nSynthetic Optimization:")
    print("  Quadratic function - Final losses:")
    for name, losses in quadratic_results.items():
        print(f"    {name}: {losses[-1]:.6f}")
    
    print("\n  Rosenbrock function - Final losses:")
    for name, losses in rosenbrock_results.items():
        print(f"    {name}: {losses[-1]:.6f}")
    
    print("\nCIFAR-10 Results:")
    for name, results in cifar10_eval.items():
        print(f"  {name}: Test Acc = {results['test_acc']:.2f}%")
    
    print("\nMNIST Ablation Study:")
    print(f"  Best configuration: {best_mnist_config[0]} with accuracy {best_mnist_config[1]['final_test_acc']:.2f}%")
    
    print("\nExperiments completed successfully!")
    print("Results and visualizations saved in the 'logs' directory")
    print("Trained models saved in the 'models' directory")

if __name__ == "__main__":
    main()
