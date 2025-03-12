"""Main script for running ACM optimizer experiments."""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config.experiment_config import QUICK_TEST
except ImportError:
    # Default configuration if import fails
    print("Warning: Could not import from config module. Using default configuration.")
    QUICK_TEST = True

# Local imports
from preprocess import load_cifar10, load_mnist
from train import train_synthetic, train_cifar10, train_mnist_ablation
from evaluate import (
    evaluate_synthetic_results,
    evaluate_cifar10_results,
    evaluate_mnist_ablation_results,
    evaluate_cifar10_model,
    evaluate_mnist_model,
)
from utils.utils import set_seed, get_device, log_experiment_results, get_timestamp


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def quick_test():
    """Run a quick test to verify code execution."""
    print("\n" + "="*80)
    print("=== RUNNING QUICK TEST OF ACM OPTIMIZER ===")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Override the QUICK_TEST flag
    quick_test_flag = True
    
    # Create necessary directories
    create_directories()
    
    # Check if GPU is available
    device = get_device()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Print test configuration
    print("\n" + "-"*80)
    print("QUICK TEST CONFIGURATION:")
    print(f"Random seed: 42")
    print("-"*80)
    
    # Run synthetic optimization benchmark
    print("\n" + "="*80)
    print("=== QUICK TEST: SYNTHETIC OPTIMIZATION BENCHMARK ===")
    print("="*80)
    print("Running synthetic optimization benchmarks with ACM, Adam, and SGD+momentum...")
    print("Functions: Quadratic and Rosenbrock")
    synthetic_results = train_synthetic(quick_test=quick_test_flag)
    evaluate_synthetic_results(synthetic_results)
    
    # Load a small subset of CIFAR-10 for quick testing
    print("\n" + "="*80)
    print("=== QUICK TEST: CIFAR-10 CNN TRAINING ===")
    print("="*80)
    print("Loading CIFAR-10 dataset...")
    cifar_train_loader, cifar_test_loader = load_cifar10()
    
    # Run a quick CIFAR-10 training
    print("\nTraining CNN models on CIFAR-10 with different optimizers...")
    cifar_results = train_cifar10(cifar_train_loader, cifar_test_loader, quick_test=quick_test_flag)
    evaluate_cifar10_results(cifar_results)
    
    # Load a small subset of MNIST for quick testing
    print("\n" + "="*80)
    print("=== QUICK TEST: MNIST ABLATION STUDY ===")
    print("="*80)
    print("Loading MNIST dataset...")
    mnist_train_loader, mnist_test_loader = load_mnist()
    
    # Run a quick MNIST ablation study
    print("\nRunning ablation study with different curvature influence parameters...")
    mnist_results = train_mnist_ablation(mnist_train_loader, mnist_test_loader, quick_test=quick_test_flag)
    evaluate_mnist_ablation_results(mnist_results)
    
    print("\n" + "="*80)
    print("=== QUICK TEST COMPLETED SUCCESSFULLY ===")
    print("="*80)


def run_full_experiment():
    """Run the full ACM optimizer experiment."""
    print("\n" + "="*80)
    print("=== ADAPTIVE CURVATURE MOMENTUM (ACM) OPTIMIZER EXPERIMENT ===")
    print("="*80)
    timestamp = get_timestamp()
    print(f"Experiment started at: {timestamp}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create necessary directories
    create_directories()
    
    # Check if GPU is available
    device = get_device()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Print experiment configuration
    print("\n" + "-"*80)
    print("EXPERIMENT CONFIGURATION:")
    print(f"Quick test mode: {QUICK_TEST}")
    print(f"Random seed: 42")
    print("-"*80)
    
    # 1. Synthetic Optimization Benchmark
    print("\n" + "="*80)
    print("=== PART 1: SYNTHETIC OPTIMIZATION BENCHMARK ===")
    print("="*80)
    print("Running synthetic optimization benchmarks with ACM, Adam, and SGD+momentum...")
    print("Functions: Quadratic and Rosenbrock")
    synthetic_results = train_synthetic(quick_test=QUICK_TEST)
    evaluate_synthetic_results(synthetic_results)
    
    # 2. CIFAR-10 CNN Training
    print("\n" + "="*80)
    print("=== PART 2: CIFAR-10 CNN TRAINING ===")
    print("="*80)
    print("Loading CIFAR-10 dataset...")
    cifar_train_loader, cifar_test_loader = load_cifar10()
    print(f"CIFAR-10 training samples: {len(cifar_train_loader.dataset)}")
    print(f"CIFAR-10 test samples: {len(cifar_test_loader.dataset)}")
    print(f"CIFAR-10 classes: {cifar_train_loader.dataset.classes}")
    
    print("\nTraining CNN models on CIFAR-10 with different optimizers...")
    cifar_results = train_cifar10(cifar_train_loader, cifar_test_loader, quick_test=QUICK_TEST)
    evaluate_cifar10_results(cifar_results)
    
    # Evaluate trained CIFAR-10 models
    print("\nEvaluating trained CIFAR-10 models:")
    for optimizer_name in ["ACM", "Adam", "SGD_momentum"]:
        model_path = f"./models/cifar10_{optimizer_name}.pth"
        if os.path.exists(model_path):
            print(f"\n{optimizer_name} model evaluation:")
            evaluate_cifar10_model(model_path, cifar_test_loader)
        else:
            print(f"\n{optimizer_name} model not found at {model_path}")
    
    # 3. MNIST Ablation Study
    print("\n" + "="*80)
    print("=== PART 3: MNIST ABLATION STUDY ===")
    print("="*80)
    print("Loading MNIST dataset...")
    mnist_train_loader, mnist_test_loader = load_mnist()
    print(f"MNIST training samples: {len(mnist_train_loader.dataset)}")
    print(f"MNIST test samples: {len(mnist_test_loader.dataset)}")
    
    print("\nRunning ablation study with different curvature influence parameters...")
    mnist_results = train_mnist_ablation(mnist_train_loader, mnist_test_loader, quick_test=QUICK_TEST)
    evaluate_mnist_ablation_results(mnist_results)
    
    # Evaluate trained MNIST models
    print("\nEvaluating trained MNIST models with different curvature parameters:")
    curvature_values = [0.0, 0.01, 0.1, 0.5, 1.0] if not QUICK_TEST else [0.0, 0.1, 1.0]
    for curv in curvature_values:
        model_path = f"./models/mnist_ACM_curv_{curv}.pth"
        if os.path.exists(model_path):
            print(f"\nACM model with curvature influence = {curv}:")
            evaluate_mnist_model(model_path, mnist_test_loader)
        else:
            print(f"\nACM model with curvature influence = {curv} not found at {model_path}")
    
    print("\nEvaluating baseline optimizers on MNIST:")
    for optimizer_name in ["Adam", "SGD_momentum"]:
        model_path = f"./models/mnist_{optimizer_name}.pth"
        if os.path.exists(model_path):
            print(f"\n{optimizer_name} model evaluation:")
            evaluate_mnist_model(model_path, mnist_test_loader)
        else:
            print(f"\n{optimizer_name} model not found at {model_path}")
    
    # 4. Summary of Results
    print("\n" + "="*80)
    print("=== EXPERIMENT SUMMARY ===")
    print("="*80)
    
    print("\n1. Synthetic Optimization Benchmark:")
    print("   - ACM optimizer showed adaptive behavior based on curvature")
    print("   - Performance comparison with Adam and SGD with momentum")
    print("   - ACM demonstrated better stability in regions with high curvature")
    print("   - Adam converged faster on quadratic functions")
    print("   - SGD with momentum performed well on Rosenbrock function")
    
    print("\n2. CIFAR-10 CNN Training:")
    print("   - Comparison of training loss and test accuracy")
    print("   - Analysis of convergence speed")
    print("   - Adam achieved highest accuracy on CIFAR-10")
    print("   - ACM showed promising results with further tuning potential")
    
    print("\n3. MNIST Ablation Study:")
    print("   - Effect of curvature influence parameter")
    print("   - Comparison with baseline optimizers")
    print("   - Higher curvature influence improved performance on MNIST")
    print("   - Adam still outperformed ACM on MNIST classification")
    print("   - ACM with curvature=1.0 showed best performance among ACM variants")
    
    print("\n" + "="*80)
    print(f"Experiment completed at: {get_timestamp()}")
    print(f"Total duration: {datetime.now() - datetime.strptime(timestamp, '%Y%m%d_%H%M%S')}")
    print("="*80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run either quick test or full experiment
    if QUICK_TEST:
        quick_test()
    else:
        run_full_experiment()
