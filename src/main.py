import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

# Fix import paths - use relative imports instead of absolute
from preprocess import load_cifar10, load_mnist
from train import SimpleCNN, MNISTNet, train_model, run_synthetic_optimization
from evaluate import evaluate_model, plot_training_curves, plot_synthetic_optimization, ablation_study_plot
from optimizers import ACMOptimizer

# Load configuration
sys.path.append("config")
try:
    from experiment_config import *
except ImportError:
    # Default configuration if import fails
    QUICK_TEST = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OPTIMIZER_CONFIGS = {
        "ACM": {"lr": 0.01, "beta": 0.9, "curvature_influence": 0.1},
        "Adam": {"lr": 0.001},
        "SGD_mom": {"lr": 0.01}
    }
    SYNTHETIC_ITERS = 100 if not QUICK_TEST else 10
    CIFAR_EPOCHS = 10 if not QUICK_TEST else 1
    MNIST_EPOCHS = 5 if not QUICK_TEST else 1
    BATCH_SIZE = 128
    NUM_WORKERS = 2
    ABLATION_PARAM = "curvature_influence"
    ABLATION_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5] if not QUICK_TEST else [0.01, 0.1]
    RESULTS_DIR = "results"

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")
    
    # Log to file as well
    with open("logs/experiment.log", "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f" {title} ".center(80, "=") + "\n")
        f.write("="*80 + "\n\n")

def quick_test():
    """Run a quick test to verify the implementation works"""
    print("Running quick test to verify implementation...")
    
    # Test synthetic optimization
    results = run_synthetic_optimization("ACM", {"lr": 0.1, "beta": 0.9, "curvature_influence": 0.1}, num_iters=10)
    print(f"Quadratic final loss: {results['quadratic'][-1]:.6f}")
    print(f"Rosenbrock final loss: {results['rosenbrock'][-1]:.6f}")
    
    # Test MNIST (smaller dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_loader, test_loader = load_mnist(batch_size=32)
    model = MNISTNet()
    
    # Only train for 2 batches
    batches_to_train = 2
    model.to(device)
    optimizer = ACMOptimizer(model.parameters(), lr=0.01, beta=0.9, curvature_influence=0.1)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= batches_to_train:
            break
            
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_idx+1} loss: {loss.item():.6f}")
    
    print("Quick test completed successfully!")
    return True

def run_synthetic_experiment():
    """Run experiments on synthetic optimization functions"""
    print_section_header("Synthetic Optimization Benchmark")
    
    results_dict = {}
    for opt_name, opt_kwargs in OPTIMIZER_CONFIGS.items():
        print(f"Running synthetic optimization with {opt_name}...")
        results = run_synthetic_optimization(opt_name, opt_kwargs, num_iters=SYNTHETIC_ITERS)
        results_dict[opt_name] = results
        print(f"  Final quadratic loss: {results['quadratic'][-1]:.6f}")
        print(f"  Final rosenbrock loss: {results['rosenbrock'][-1]:.6f}")
        print(f"  Final quadratic position: {results['final_x_quad']}")
        print(f"  Final rosenbrock position: {results['final_x_rosen']}")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Plot results
    fig = plot_synthetic_optimization(results_dict, save_path="results/synthetic_optimization.png")
    plt.close(fig)
    
    return results_dict

def run_cifar10_experiment():
    """Run experiments on CIFAR-10 dataset"""
    print_section_header("CIFAR-10 Classification Experiment")
    
    device = DEVICE
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    results_dict = {}
    for opt_name, opt_kwargs in OPTIMIZER_CONFIGS.items():
        print(f"Training with {opt_name} optimizer...")
        model = SimpleCNN()
        
        start_time = time.time()
        train_losses, test_accuracies = train_model(
            model, train_loader, test_loader, opt_name, opt_kwargs,
            device, epochs=CIFAR_EPOCHS, save_path=f"models/cifar10_{opt_name}.pt"
        )
        elapsed_time = time.time() - start_time
        
        results_dict[opt_name] = {
            'train_loss': train_losses,
            'test_accuracy': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'training_time': elapsed_time
        }
        
        print(f"  Final test accuracy: {test_accuracies[-1]:.4f}")
        print(f"  Training time: {elapsed_time:.2f} seconds")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Plot results
    fig = plot_training_curves(results_dict, title="CIFAR-10", save_path="results/cifar10_training.png")
    plt.close(fig)
    
    return results_dict

def run_mnist_experiment():
    """Run experiments on MNIST dataset"""
    print_section_header("MNIST Classification Experiment")
    
    device = DEVICE
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    results_dict = {}
    for opt_name, opt_kwargs in OPTIMIZER_CONFIGS.items():
        print(f"Training with {opt_name} optimizer...")
        model = MNISTNet()
        
        start_time = time.time()
        train_losses, test_accuracies = train_model(
            model, train_loader, test_loader, opt_name, opt_kwargs,
            device, epochs=MNIST_EPOCHS, save_path=f"models/mnist_{opt_name}.pt"
        )
        elapsed_time = time.time() - start_time
        
        results_dict[opt_name] = {
            'train_loss': train_losses,
            'test_accuracy': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'training_time': elapsed_time
        }
        
        print(f"  Final test accuracy: {test_accuracies[-1]:.4f}")
        print(f"  Training time: {elapsed_time:.2f} seconds")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Plot results
    fig = plot_training_curves(results_dict, title="MNIST", save_path="results/mnist_training.png")
    plt.close(fig)
    
    return results_dict

def run_ablation_study():
    """Run ablation study on the curvature_influence parameter of ACM optimizer"""
    print_section_header(f"Ablation Study: Effect of {ABLATION_PARAM}")
    
    device = DEVICE
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset for ablation study...")
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    results_dict = {}
    for param_value in ABLATION_VALUES:
        print(f"Testing {ABLATION_PARAM}={param_value}...")
        model = MNISTNet()
        
        # Set the parameter value for this run
        optimizer_kwargs = OPTIMIZER_CONFIGS["ACM"].copy()
        optimizer_kwargs[ABLATION_PARAM] = param_value
        
        train_losses, test_accuracies = train_model(
            model, train_loader, test_loader, "ACM", optimizer_kwargs,
            device, epochs=MNIST_EPOCHS, save_path=None
        )
        
        results_dict[param_value] = {
            'train_loss': train_losses,
            'test_accuracy': test_accuracies,
            'final_accuracy': test_accuracies[-1]
        }
        
        print(f"  Final test accuracy: {test_accuracies[-1]:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Plot results
    fig = ablation_study_plot(results_dict, ABLATION_PARAM, save_path=f"results/ablation_{ABLATION_PARAM}.png")
    plt.close(fig)
    
    return results_dict

def main():
    """Main function to run all experiments"""
    # Initialize log file
    os.makedirs("logs", exist_ok=True)
    with open("logs/experiment.log", "w") as f:
        f.write(f"ACM Optimizer Experiments - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    print_section_header("Adaptive Curvature Momentum (ACM) Optimizer Experiments")
    print(f"Running on device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Display experiment configuration
    print("\nExperiment Configuration:")
    print(f"  Quick test mode: {QUICK_TEST}")
    print(f"  Synthetic iterations: {SYNTHETIC_ITERS}")
    print(f"  CIFAR-10 epochs: {CIFAR_EPOCHS}")
    print(f"  MNIST epochs: {MNIST_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Optimizers: {', '.join(OPTIMIZER_CONFIGS.keys())}")
    
    # Run a quick test first to verify implementation
    if quick_test():
        print("Quick test passed, proceeding with experiments...")
    else:
        print("Quick test failed, exiting...")
        return
    
    # Record start time
    start_time = time.time()
    
    # Run synthetic optimization experiments
    synthetic_results = run_synthetic_experiment()
    
    # Run CIFAR-10 experiments
    cifar10_results = run_cifar10_experiment()
    
    # Run MNIST experiments
    mnist_results = run_mnist_experiment()
    
    # Run ablation study
    ablation_results = run_ablation_study()
    
    # Calculate total experiment time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_section_header("Experiment Summary")
    print(f"Total experiment time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Synthetic optimization summary
    print("\nSynthetic Optimization Final Losses:")
    for opt_name in OPTIMIZER_CONFIGS.keys():
        print(f"  {opt_name}:")
        print(f"    Quadratic: {synthetic_results[opt_name]['quadratic'][-1]:.6f}")
        print(f"    Rosenbrock: {synthetic_results[opt_name]['rosenbrock'][-1]:.6f}")
        print(f"    Final quadratic position: {synthetic_results[opt_name]['final_x_quad']}")
        print(f"    Final rosenbrock position: {synthetic_results[opt_name]['final_x_rosen']}")
    
    # CIFAR-10 summary
    print("\nCIFAR-10 Final Test Accuracies:")
    for opt_name in OPTIMIZER_CONFIGS.keys():
        print(f"  {opt_name}: {cifar10_results[opt_name]['final_accuracy']:.4f} (training time: {cifar10_results[opt_name]['training_time']:.2f}s)")
    
    # MNIST summary
    print("\nMNIST Final Test Accuracies:")
    for opt_name in OPTIMIZER_CONFIGS.keys():
        print(f"  {opt_name}: {mnist_results[opt_name]['final_accuracy']:.4f} (training time: {mnist_results[opt_name]['training_time']:.2f}s)")
    
    # Ablation study summary
    print(f"\nAblation Study Results ({ABLATION_PARAM}):")
    for param_value in sorted(ABLATION_VALUES):
        print(f"  {param_value}: {ablation_results[param_value]['final_accuracy']:.4f}")
    
    # Performance comparison
    print("\nPerformance Comparison:")
    best_cifar = max(cifar10_results.items(), key=lambda x: x[1]['final_accuracy'])
    best_mnist = max(mnist_results.items(), key=lambda x: x[1]['final_accuracy'])
    print(f"  Best CIFAR-10 optimizer: {best_cifar[0]} with accuracy {best_cifar[1]['final_accuracy']:.4f}")
    print(f"  Best MNIST optimizer: {best_mnist[0]} with accuracy {best_mnist[1]['final_accuracy']:.4f}")
    
    # Log final results to file
    with open("logs/experiment.log", "a") as f:
        f.write(f"\nExperiment completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiment time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n\n")
        f.write("Summary of Results:\n")
        f.write(f"Best CIFAR-10 optimizer: {best_cifar[0]} with accuracy {best_cifar[1]['final_accuracy']:.4f}\n")
        f.write(f"Best MNIST optimizer: {best_mnist[0]} with accuracy {best_mnist[1]['final_accuracy']:.4f}\n")
    
    print("\nExperiments completed successfully!")
    print("Detailed results and plots saved to 'results/' directory")
    print("Log file saved to 'logs/experiment.log'")

if __name__ == "__main__":
    main()
