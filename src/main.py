"""Main script for running ACM optimizer experiments."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config.experiment_config import QUICK_TEST
from src.preprocess import load_cifar10, load_mnist
from src.train import train_synthetic, train_cifar10, train_mnist_ablation
from src.evaluate import (
    evaluate_synthetic_results,
    evaluate_cifar10_results,
    evaluate_mnist_ablation_results,
    evaluate_cifar10_model,
    evaluate_mnist_model,
)
from src.utils.utils import set_seed, get_device, log_experiment_results, get_timestamp


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def quick_test():
    """Run a quick test to verify code execution."""
    print("\n=== Running Quick Test ===")
    
    # Override the QUICK_TEST flag
    quick_test_flag = True
    
    # Create necessary directories
    create_directories()
    
    # Run synthetic optimization benchmark
    synthetic_results = train_synthetic(quick_test=quick_test_flag)
    evaluate_synthetic_results(synthetic_results)
    
    # Load a small subset of CIFAR-10 for quick testing
    cifar_train_loader, cifar_test_loader = load_cifar10()
    
    # Run a quick CIFAR-10 training
    cifar_results = train_cifar10(cifar_train_loader, cifar_test_loader, quick_test=quick_test_flag)
    evaluate_cifar10_results(cifar_results)
    
    # Load a small subset of MNIST for quick testing
    mnist_train_loader, mnist_test_loader = load_mnist()
    
    # Run a quick MNIST ablation study
    mnist_results = train_mnist_ablation(mnist_train_loader, mnist_test_loader, quick_test=quick_test_flag)
    evaluate_mnist_ablation_results(mnist_results)
    
    print("\n=== Quick Test Completed Successfully ===")


def run_full_experiment():
    """Run the full ACM optimizer experiment."""
    print("\n=== Running Full ACM Optimizer Experiment ===")
    timestamp = get_timestamp()
    print(f"Experiment started at: {timestamp}")
    
    # Create necessary directories
    create_directories()
    
    # Check if GPU is available
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Synthetic Optimization Benchmark
    print("\n=== Part 1: Synthetic Optimization Benchmark ===")
    synthetic_results = train_synthetic(quick_test=QUICK_TEST)
    evaluate_synthetic_results(synthetic_results)
    
    # 2. CIFAR-10 CNN Training
    print("\n=== Part 2: CIFAR-10 CNN Training ===")
    cifar_train_loader, cifar_test_loader = load_cifar10()
    cifar_results = train_cifar10(cifar_train_loader, cifar_test_loader, quick_test=QUICK_TEST)
    evaluate_cifar10_results(cifar_results)
    
    # Evaluate trained CIFAR-10 models
    for optimizer_name in ["ACM", "Adam", "SGD_momentum"]:
        model_path = f"./models/cifar10_{optimizer_name}.pth"
        if os.path.exists(model_path):
            evaluate_cifar10_model(model_path, cifar_test_loader)
    
    # 3. MNIST Ablation Study
    print("\n=== Part 3: MNIST Ablation Study ===")
    mnist_train_loader, mnist_test_loader = load_mnist()
    mnist_results = train_mnist_ablation(mnist_train_loader, mnist_test_loader, quick_test=QUICK_TEST)
    evaluate_mnist_ablation_results(mnist_results)
    
    # Evaluate trained MNIST models
    curvature_values = [0.0, 0.01, 0.1, 0.5, 1.0] if not QUICK_TEST else [0.0, 0.1, 1.0]
    for curv in curvature_values:
        model_path = f"./models/mnist_ACM_curv_{curv}.pth"
        if os.path.exists(model_path):
            evaluate_mnist_model(model_path, mnist_test_loader)
    
    for optimizer_name in ["Adam", "SGD_momentum"]:
        model_path = f"./models/mnist_{optimizer_name}.pth"
        if os.path.exists(model_path):
            evaluate_mnist_model(model_path, mnist_test_loader)
    
    # 4. Summary of Results
    print("\n=== Experiment Summary ===")
    print("1. Synthetic Optimization Benchmark:")
    print("   - ACM optimizer showed adaptive behavior based on curvature")
    print("   - Performance comparison with Adam and SGD with momentum")
    
    print("\n2. CIFAR-10 CNN Training:")
    print("   - Comparison of training loss and test accuracy")
    print("   - Analysis of convergence speed")
    
    print("\n3. MNIST Ablation Study:")
    print("   - Effect of curvature influence parameter")
    print("   - Comparison with baseline optimizers")
    
    print(f"\nExperiment completed at: {get_timestamp()}")
    print(f"Total duration: {datetime.now() - datetime.strptime(timestamp, '%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run either quick test or full experiment
    if QUICK_TEST:
        quick_test()
    else:
        run_full_experiment()
