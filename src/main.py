"""
Main script for running ACM optimizer experiments.

This script orchestrates the entire process from data preprocessing to model training
and evaluation for the following experiments:
1. Real-World Convergence Experiment (using CIFAR-10 and ResNet-18)
2. Synthetic Loss Landscape Experiment (using quadratic and Rosenbrock functions)
3. Hyperparameter Sensitivity and Robustness Analysis (using a simple CNN)
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
import time
from datetime import datetime

# Set matplotlib backend to Agg for non-interactive plotting (useful for servers without display)
matplotlib.use('Agg')

# Import modules from the project
from preprocess import load_cifar10, prepare_synthetic_functions, get_initial_points
from train import train_resnet_cifar10, optimize_synthetic_function, train_cnn_hyperparameter_search
from evaluate import evaluate_real_world_experiment, evaluate_synthetic_experiment, evaluate_hyperparameter_experiment
from utils.optimizer import ACM

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import RANDOM_SEED, DEVICE, TEST_MODE
from config.experiment_config import REAL_WORLD_CONFIG, SYNTHETIC_CONFIG, HYPERPARAMETER_CONFIG

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use GPU if available
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for logs and models
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

def run_real_world_experiment(test_run=False):
    """
    Run the real-world convergence experiment using CIFAR-10 and ResNet-18.
    
    Args:
        test_run (bool): If True, use a small subset of data and fewer epochs for quick testing
    """
    print("\n" + "="*80)
    print("Running Real-World Convergence Experiment (CIFAR-10 + ResNet-18)")
    print("="*80)
    
    # Get configuration
    config = REAL_WORLD_CONFIG
    
    # Adjust for test run
    if test_run:
        print("Running in TEST mode with reduced dataset and epochs")
        num_epochs = config["test_epochs"]
    else:
        num_epochs = config["num_epochs"]
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(test_run=test_run)
    
    # Initialize results dictionary
    results = {}
    
    # Train with each optimizer
    for opt_name, opt_params in config["optimizers"].items():
        print(f"\nTraining ResNet-18 with {opt_name} optimizer...")
        print(f"Parameters: {opt_params}")
        
        # Train model
        model, history = train_resnet_cifar10(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer_name=opt_name,
            optimizer_params=opt_params,
            test_run=test_run
        )
        
        # Save model
        model_path = f"models/resnet18_{opt_name.lower()}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Store results
        results[opt_name] = history
    
    # Evaluate results
    metrics = evaluate_real_world_experiment(results)
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"logs/real_world_metrics_{timestamp}.txt"
    
    with open(metrics_path, "w") as f:
        f.write("Real-World Convergence Experiment Metrics\n")
        f.write("="*50 + "\n\n")
        
        for opt_name, opt_metrics in metrics.items():
            f.write(f"{opt_name} Optimizer:\n")
            for metric_name, metric_value in opt_metrics.items():
                if isinstance(metric_value, float):
                    f.write(f"  {metric_name}: {metric_value:.4f}\n")
                else:
                    f.write(f"  {metric_name}: {metric_value}\n")
            f.write("\n")
    
    print(f"Metrics saved to {metrics_path}")
    return results

def run_synthetic_experiment(test_run=False):
    """
    Run the synthetic loss landscape experiment using quadratic and Rosenbrock functions.
    
    Args:
        test_run (bool): If True, use fewer iterations for quick testing
    """
    print("\n" + "="*80)
    print("Running Synthetic Loss Landscape Experiment")
    print("="*80)
    
    # Get configuration
    config = SYNTHETIC_CONFIG
    
    # Prepare synthetic functions
    print("\nPreparing synthetic functions...")
    quadratic_fn, rosenbrock_fn = prepare_synthetic_functions()
    x_init_quad, x_init_rosen = get_initial_points()
    
    # Move tensors to device
    x_init_quad = x_init_quad.to(device)
    x_init_rosen = x_init_rosen.to(device)
    
    # Results dictionary
    results = {
        "quadratic": {},
        "rosenbrock": {}
    }
    
    # Quadratic function optimization
    print("\nOptimizing quadratic function...")
    
    # Set iterations based on test mode
    quad_iters = config["test_iterations_quadratic"] if test_run else config["quadratic"]["iterations"]
    
    # ACM optimization
    print("Using ACM optimizer...")
    acm_trajectory, acm_lr_evolution = optimize_synthetic_function(
        fn=lambda x: quadratic_fn(x, A=torch.tensor(config["quadratic"]["matrix"], device=device)),
        x_init=x_init_quad.clone(),
        optimizer_type="acm",
        alpha=config["quadratic"]["alpha"],
        beta=config["quadratic"]["beta"],
        n_iters=quad_iters,
        test_run=test_run
    )
    
    # SGD optimization
    print("Using SGD optimizer...")
    sgd_trajectory, _ = optimize_synthetic_function(
        fn=lambda x: quadratic_fn(x, A=torch.tensor(config["quadratic"]["matrix"], device=device)),
        x_init=x_init_quad.clone(),
        optimizer_type="sgd",
        alpha=config["quadratic"]["alpha"],
        n_iters=quad_iters,
        test_run=test_run
    )
    
    # Store results
    results["quadratic"]["acm_trajectory"] = acm_trajectory
    results["quadratic"]["sgd_trajectory"] = sgd_trajectory
    results["quadratic"]["acm_lr_evolution"] = acm_lr_evolution
    
    # Rosenbrock function optimization
    print("\nOptimizing Rosenbrock function...")
    
    # Set iterations based on test mode
    rosen_iters = config["test_iterations_rosenbrock"] if test_run else config["rosenbrock"]["iterations"]
    
    # ACM optimization
    print("Using ACM optimizer...")
    acm_trajectory, acm_lr_evolution = optimize_synthetic_function(
        fn=lambda x: rosenbrock_fn(x, a=config["rosenbrock"]["a"], b=config["rosenbrock"]["b"]),
        x_init=x_init_rosen.clone(),
        optimizer_type="acm",
        alpha=config["rosenbrock"]["alpha"],
        beta=config["rosenbrock"]["beta"],
        n_iters=rosen_iters,
        test_run=test_run
    )
    
    # SGD optimization
    print("Using SGD optimizer...")
    sgd_trajectory, _ = optimize_synthetic_function(
        fn=lambda x: rosenbrock_fn(x, a=config["rosenbrock"]["a"], b=config["rosenbrock"]["b"]),
        x_init=x_init_rosen.clone(),
        optimizer_type="sgd",
        alpha=config["rosenbrock"]["alpha"],
        n_iters=rosen_iters,
        test_run=test_run
    )
    
    # Store results
    results["rosenbrock"]["acm_trajectory"] = acm_trajectory
    results["rosenbrock"]["sgd_trajectory"] = sgd_trajectory
    results["rosenbrock"]["acm_lr_evolution"] = acm_lr_evolution
    
    # Evaluate results
    metrics = evaluate_synthetic_experiment(results)
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"logs/synthetic_metrics_{timestamp}.txt"
    
    with open(metrics_path, "w") as f:
        f.write("Synthetic Loss Landscape Experiment Metrics\n")
        f.write("="*50 + "\n\n")
        
        f.write("Quadratic Function Optimization:\n")
        for metric_name, metric_value in metrics["quadratic"].items():
            if isinstance(metric_value, float):
                f.write(f"  {metric_name}: {metric_value:.6f}\n")
            else:
                f.write(f"  {metric_name}: {metric_value}\n")
        
        f.write("\nRosenbrock Function Optimization:\n")
        for metric_name, metric_value in metrics["rosenbrock"].items():
            if isinstance(metric_value, float):
                f.write(f"  {metric_name}: {metric_value:.6f}\n")
            else:
                f.write(f"  {metric_name}: {metric_value}\n")
    
    print(f"Metrics saved to {metrics_path}")
    return results

def run_hyperparameter_experiment(test_run=False):
    """
    Run the hyperparameter sensitivity experiment using a simple CNN on CIFAR-10.
    
    Args:
        test_run (bool): If True, use a small subset of data and fewer epochs for quick testing
    """
    print("\n" + "="*80)
    print("Running Hyperparameter Sensitivity Experiment")
    print("="*80)
    
    # Get configuration
    config = HYPERPARAMETER_CONFIG
    
    # Adjust for test run
    if test_run:
        print("Running in TEST mode with reduced dataset and epochs")
        num_epochs = config["test_epochs"]
    else:
        num_epochs = config["num_epochs"]
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(test_run=test_run)
    
    # Results dictionary
    results = {
        "ACM": [],
        "Adam": []
    }
    
    # ACM hyperparameter grid search
    print("\nRunning ACM hyperparameter grid search...")
    acm_results = train_cnn_hyperparameter_search(
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_type="ACM",
        param_grid=config["acm_grid"],
        test_run=test_run
    )
    results["ACM"] = acm_results
    
    # Adam hyperparameter grid search
    print("\nRunning Adam hyperparameter grid search...")
    adam_results = train_cnn_hyperparameter_search(
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_type="Adam",
        param_grid=config["adam_grid"],
        test_run=test_run
    )
    results["Adam"] = adam_results
    
    # Evaluate results
    metrics = evaluate_hyperparameter_experiment(results)
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"logs/hyperparameter_metrics_{timestamp}.txt"
    
    with open(metrics_path, "w") as f:
        f.write("Hyperparameter Sensitivity Experiment Metrics\n")
        f.write("="*50 + "\n\n")
        
        f.write("Best ACM Configuration:\n")
        for param_name, param_value in metrics["best_acm"].items():
            if isinstance(param_value, float):
                f.write(f"  {param_name}: {param_value:.4f}\n")
            else:
                f.write(f"  {param_name}: {param_value}\n")
        
        f.write("\nBest Adam Configuration:\n")
        for param_name, param_value in metrics["best_adam"].items():
            if isinstance(param_value, float):
                f.write(f"  {param_name}: {param_value:.4f}\n")
            else:
                f.write(f"  {param_name}: {param_value}\n")
        
        f.write(f"\nACM Robustness (Std Dev): {metrics['acm_robustness']:.4f}\n")
        f.write(f"Adam Robustness (Std Dev): {metrics['adam_robustness']:.4f}\n")
    
    print(f"Metrics saved to {metrics_path}")
    return results

def test_all_experiments():
    """
    Run all experiments in test mode for quick verification.
    """
    print("\n" + "="*80)
    print("Running Quick Test of All Experiments")
    print("="*80)
    
    start_time = time.time()
    
    # Run experiments in test mode
    run_real_world_experiment(test_run=True)
    run_synthetic_experiment(test_run=True)
    run_hyperparameter_experiment(test_run=True)
    
    total_time = time.time() - start_time
    print(f"\nAll tests completed in {total_time:.2f} seconds")

def main():
    """
    Main function to parse arguments and run experiments.
    """
    parser = argparse.ArgumentParser(description="ACM Optimizer Experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "real_world", "synthetic", "hyperparameter", "test"],
                        help="Experiment to run")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with reduced dataset and epochs")
    
    args = parser.parse_args()
    
    # Set test mode
    test_run = args.test or TEST_MODE
    
    # Print system information
    print("\n" + "="*80)
    print("ACM Optimizer Experiments")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Running in {'TEST' if test_run else 'FULL'} mode")
    print("="*80)
    
    # Print experiment description
    print("\nAdaptive Curvature Momentum (ACM) Optimizer")
    print("="*50)
    print("The ACM optimizer utilizes local quadratic approximations to adaptively")
    print("adjust the update direction and scale based on curvature information.")
    print("\nKey Features:")
    print("✅ Combines Adam-style adaptability with curvature-aware updates")
    print("✅ Faster convergence in flat regions, careful steps in sharp valleys")
    print("✅ Hessian-free approximation with low computational overhead")
    print("✅ Suitable for large-scale models such as ResNets and Transformers")
    print("\nExperiments:")
    print("1. Real-World Convergence: CIFAR-10 + ResNet-18")
    print("2. Synthetic Loss Landscape: Quadratic and Rosenbrock functions")
    print("3. Hyperparameter Sensitivity: Grid search analysis")
    print("="*80)
    
    # Run selected experiment
    if args.experiment == "all":
        if test_run:
            test_all_experiments()
        else:
            run_real_world_experiment(test_run=test_run)
            run_synthetic_experiment(test_run=test_run)
            run_hyperparameter_experiment(test_run=test_run)
    elif args.experiment == "real_world":
        run_real_world_experiment(test_run=test_run)
    elif args.experiment == "synthetic":
        run_synthetic_experiment(test_run=test_run)
    elif args.experiment == "hyperparameter":
        run_hyperparameter_experiment(test_run=test_run)
    elif args.experiment == "test":
        test_all_experiments()

if __name__ == "__main__":
    main()
