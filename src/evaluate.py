import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# Use relative imports when running from within the package
try:
    # When running as a module (e.g., python -m src.evaluate)
    from src.train import SimpleCNN, MNISTNet, quadratic_loss, rosenbrock_loss
except ModuleNotFoundError:
    # When running directly (e.g., python src/evaluate.py)
    from train import SimpleCNN, MNISTNet, quadratic_loss, rosenbrock_loss

def evaluate_synthetic(results, config):
    """
    Evaluate and visualize results from synthetic optimization problems.
    
    Args:
        results: Dictionary containing training results for different optimizers
        config: Configuration dictionary with evaluation parameters
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Evaluating synthetic optimization results...")
    
    # Create directory for saving plots
    os.makedirs('logs', exist_ok=True)
    
    # Plot quadratic function optimization results
    plt.figure(figsize=(10, 6))
    for optimizer_name, optimizer_results in results.items():
        plt.plot(optimizer_results['quadratic']['losses'], label=optimizer_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Quadratic Function Optimization')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Save plot
    plt.savefig('logs/quadratic_optimization.png')
    print(f"Saved quadratic optimization plot to logs/quadratic_optimization.png")
    
    # Plot Rosenbrock function optimization results
    plt.figure(figsize=(10, 6))
    for optimizer_name, optimizer_results in results.items():
        plt.plot(optimizer_results['rosenbrock']['losses'], label=optimizer_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Rosenbrock Function Optimization')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Save plot
    plt.savefig('logs/rosenbrock_optimization.png')
    print(f"Saved Rosenbrock optimization plot to logs/rosenbrock_optimization.png")
    
    # Calculate convergence metrics
    metrics = {}
    for optimizer_name, optimizer_results in results.items():
        # Calculate final loss values
        final_quadratic_loss = optimizer_results['quadratic']['losses'][-1]
        final_rosenbrock_loss = optimizer_results['rosenbrock']['losses'][-1]
        
        # Calculate convergence rates (average loss reduction per iteration)
        quadratic_losses = optimizer_results['quadratic']['losses']
        rosenbrock_losses = optimizer_results['rosenbrock']['losses']
        
        # Skip first iteration for convergence rate calculation
        if len(quadratic_losses) > 10:
            quadratic_convergence_rate = (quadratic_losses[0] - quadratic_losses[-1]) / len(quadratic_losses)
            rosenbrock_convergence_rate = (rosenbrock_losses[0] - rosenbrock_losses[-1]) / len(rosenbrock_losses)
        else:
            quadratic_convergence_rate = 0
            rosenbrock_convergence_rate = 0
        
        metrics[optimizer_name] = {
            'quadratic': {
                'final_loss': final_quadratic_loss,
                'convergence_rate': quadratic_convergence_rate
            },
            'rosenbrock': {
                'final_loss': final_rosenbrock_loss,
                'convergence_rate': rosenbrock_convergence_rate
            }
        }
    
    # Print evaluation metrics
    print("\nSynthetic Optimization Metrics:")
    print("-" * 50)
    print(f"{'Optimizer':<10} | {'Quadratic Final Loss':<20} | {'Rosenbrock Final Loss':<20}")
    print("-" * 50)
    
    for optimizer_name, optimizer_metrics in metrics.items():
        quadratic_loss = optimizer_metrics['quadratic']['final_loss']
        rosenbrock_loss = optimizer_metrics['rosenbrock']['final_loss']
        print(f"{optimizer_name:<10} | {quadratic_loss:<20.6f} | {rosenbrock_loss:<20.6f}")
    
    return metrics

def evaluate_cifar10(results, config):
    """
    Evaluate and visualize results from CIFAR-10 training.
    
    Args:
        results: Dictionary containing training results for different optimizers
        config: Configuration dictionary with evaluation parameters
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Evaluating CIFAR-10 training results...")
    
    # Create directory for saving plots
    os.makedirs('logs', exist_ok=True)
    
    # Plot training and test losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for optimizer_name, optimizer_results in results.items():
        plt.plot(optimizer_results['results']['train_losses'], label=f"{optimizer_name} Train")
        plt.plot(optimizer_results['results']['test_losses'], label=f"{optimizer_name} Test", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CIFAR-10 Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training and test accuracies
    plt.subplot(1, 2, 2)
    for optimizer_name, optimizer_results in results.items():
        plt.plot(optimizer_results['results']['train_accs'], label=f"{optimizer_name} Train")
        plt.plot(optimizer_results['results']['test_accs'], label=f"{optimizer_name} Test", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CIFAR-10 Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('logs/cifar10_training.png')
    print(f"Saved CIFAR-10 training plot to logs/cifar10_training.png")
    
    # Calculate evaluation metrics
    metrics = {}
    for optimizer_name, optimizer_results in results.items():
        # Get final metrics
        final_train_loss = optimizer_results['results']['train_losses'][-1]
        final_test_loss = optimizer_results['results']['test_losses'][-1]
        final_train_acc = optimizer_results['results']['train_accs'][-1]
        final_test_acc = optimizer_results['results']['test_accs'][-1]
        
        # Calculate convergence speed (epochs to reach 90% of final accuracy)
        train_accs = optimizer_results['results']['train_accs']
        target_acc = 0.9 * final_train_acc
        
        epochs_to_converge = len(train_accs)  # Default to max epochs
        for i, acc in enumerate(train_accs):
            if acc >= target_acc:
                epochs_to_converge = i + 1
                break
        
        metrics[optimizer_name] = {
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'final_train_acc': final_train_acc,
            'final_test_acc': final_test_acc,
            'epochs_to_converge': epochs_to_converge
        }
    
    # Print evaluation metrics
    print("\nCIFAR-10 Evaluation Metrics:")
    print("-" * 80)
    print(f"{'Optimizer':<10} | {'Test Acc (%)':<12} | {'Test Loss':<10} | {'Train Acc (%)':<12} | {'Train Loss':<10} | {'Epochs to Converge':<18}")
    print("-" * 80)
    
    for optimizer_name, optimizer_metrics in metrics.items():
        print(f"{optimizer_name:<10} | {optimizer_metrics['final_test_acc']:<12.2f} | {optimizer_metrics['final_test_loss']:<10.4f} | "
              f"{optimizer_metrics['final_train_acc']:<12.2f} | {optimizer_metrics['final_train_loss']:<10.4f} | {optimizer_metrics['epochs_to_converge']:<18}")
    
    return metrics

def evaluate_mnist(results, config):
    """
    Evaluate and visualize results from MNIST training.
    
    Args:
        results: Dictionary containing training results for different optimizers
        config: Configuration dictionary with evaluation parameters
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Evaluating MNIST training results...")
    
    # Create directory for saving plots
    os.makedirs('logs', exist_ok=True)
    
    # Plot training and test losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for optimizer_name, optimizer_results in results.items():
        plt.plot(optimizer_results['results']['train_losses'], label=f"{optimizer_name} Train")
        plt.plot(optimizer_results['results']['test_losses'], label=f"{optimizer_name} Test", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MNIST Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training and test accuracies
    plt.subplot(1, 2, 2)
    for optimizer_name, optimizer_results in results.items():
        plt.plot(optimizer_results['results']['train_accs'], label=f"{optimizer_name} Train")
        plt.plot(optimizer_results['results']['test_accs'], label=f"{optimizer_name} Test", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('MNIST Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('logs/mnist_training.png')
    print(f"Saved MNIST training plot to logs/mnist_training.png")
    
    # Calculate evaluation metrics
    metrics = {}
    for optimizer_name, optimizer_results in results.items():
        # Get final metrics
        final_train_loss = optimizer_results['results']['train_losses'][-1]
        final_test_loss = optimizer_results['results']['test_losses'][-1]
        final_train_acc = optimizer_results['results']['train_accs'][-1]
        final_test_acc = optimizer_results['results']['test_accs'][-1]
        
        # Calculate convergence speed (epochs to reach 90% of final accuracy)
        train_accs = optimizer_results['results']['train_accs']
        target_acc = 0.9 * final_train_acc
        
        epochs_to_converge = len(train_accs)  # Default to max epochs
        for i, acc in enumerate(train_accs):
            if acc >= target_acc:
                epochs_to_converge = i + 1
                break
        
        metrics[optimizer_name] = {
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'final_train_acc': final_train_acc,
            'final_test_acc': final_test_acc,
            'epochs_to_converge': epochs_to_converge
        }
    
    # Print evaluation metrics
    print("\nMNIST Evaluation Metrics:")
    print("-" * 80)
    print(f"{'Optimizer':<10} | {'Test Acc (%)':<12} | {'Test Loss':<10} | {'Train Acc (%)':<12} | {'Train Loss':<10} | {'Epochs to Converge':<18}")
    print("-" * 80)
    
    for optimizer_name, optimizer_metrics in metrics.items():
        print(f"{optimizer_name:<10} | {optimizer_metrics['final_test_acc']:<12.2f} | {optimizer_metrics['final_test_loss']:<10.4f} | "
              f"{optimizer_metrics['final_train_acc']:<12.2f} | {optimizer_metrics['final_train_loss']:<10.4f} | {optimizer_metrics['epochs_to_converge']:<18}")
    
    return metrics

def evaluate_ablation_study(results, config):
    """
    Evaluate and visualize results from the ablation study.
    
    Args:
        results: Dictionary containing ablation study results
        config: Configuration dictionary with evaluation parameters
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Evaluating ablation study results...")
    
    # Create directory for saving plots
    os.makedirs('logs', exist_ok=True)
    
    # Plot learning rate study results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for result in results['lr_study']:
        plt.plot(result['losses'], label=f"LR={result['lr']}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Effect of Learning Rate on Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot beta study results
    plt.subplot(1, 3, 2)
    for result in results['beta_study']:
        plt.plot(result['losses'], label=f"Beta={result['beta']}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Effect of Beta on Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot curvature influence study results
    plt.subplot(1, 3, 3)
    for result in results['curvature_study']:
        plt.plot(result['losses'], label=f"CI={result['curvature_influence']}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Effect of Curvature Influence on Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('logs/ablation_study.png')
    print(f"Saved ablation study plot to logs/ablation_study.png")
    
    # Calculate best hyperparameters
    best_lr = min(results['lr_study'], key=lambda x: x['losses'][-1])['lr']
    best_beta = min(results['beta_study'], key=lambda x: x['losses'][-1])['beta']
    best_ci = min(results['curvature_study'], key=lambda x: x['losses'][-1])['curvature_influence']
    
    print("\nAblation Study Results:")
    print(f"Best learning rate: {best_lr}")
    print(f"Best beta value: {best_beta}")
    print(f"Best curvature influence value: {best_ci}")
    
    return {
        'best_lr': best_lr,
        'best_beta': best_beta,
        'best_curvature_influence': best_ci
    }

def quick_test():
    """Run a quick test with minimal iterations to verify code execution."""
    import torch
    import numpy as np
    
    print("Running quick evaluation test...")
    
    # Create synthetic results
    synthetic_results = {
        'ACM': {
            'quadratic': {'losses': [10.0, 8.0, 6.0, 4.0, 2.0]},
            'rosenbrock': {'losses': [100.0, 80.0, 60.0, 40.0, 20.0]}
        },
        'Adam': {
            'quadratic': {'losses': [10.0, 7.5, 5.0, 2.5, 1.0]},
            'rosenbrock': {'losses': [100.0, 75.0, 50.0, 25.0, 10.0]}
        },
        'SGD_mom': {
            'quadratic': {'losses': [10.0, 9.0, 8.0, 7.0, 6.0]},
            'rosenbrock': {'losses': [100.0, 90.0, 80.0, 70.0, 60.0]}
        }
    }
    
    # Create CIFAR-10 results
    cifar10_results = {
        'ACM': {
            'results': {
                'train_losses': [2.0, 1.8, 1.6],
                'test_losses': [2.1, 1.9, 1.7],
                'train_accs': [40.0, 50.0, 60.0],
                'test_accs': [38.0, 48.0, 58.0]
            }
        },
        'Adam': {
            'results': {
                'train_losses': [2.0, 1.7, 1.4],
                'test_losses': [2.1, 1.8, 1.5],
                'train_accs': [40.0, 55.0, 65.0],
                'test_accs': [38.0, 53.0, 63.0]
            }
        },
        'SGD_mom': {
            'results': {
                'train_losses': [2.0, 1.9, 1.8],
                'test_losses': [2.1, 2.0, 1.9],
                'train_accs': [40.0, 45.0, 50.0],
                'test_accs': [38.0, 43.0, 48.0]
            }
        }
    }
    
    # Create MNIST results
    mnist_results = {
        'ACM': {
            'results': {
                'train_losses': [0.5, 0.4, 0.3],
                'test_losses': [0.55, 0.45, 0.35],
                'train_accs': [85.0, 90.0, 95.0],
                'test_accs': [84.0, 89.0, 94.0]
            }
        },
        'Adam': {
            'results': {
                'train_losses': [0.5, 0.35, 0.25],
                'test_losses': [0.55, 0.4, 0.3],
                'train_accs': [85.0, 92.0, 96.0],
                'test_accs': [84.0, 91.0, 95.0]
            }
        },
        'SGD_mom': {
            'results': {
                'train_losses': [0.5, 0.45, 0.4],
                'test_losses': [0.55, 0.5, 0.45],
                'train_accs': [85.0, 88.0, 91.0],
                'test_accs': [84.0, 87.0, 90.0]
            }
        }
    }
    
    # Create ablation study results
    ablation_results = {
        'lr_study': [
            {'lr': 0.001, 'losses': [0.5, 0.45, 0.4]},
            {'lr': 0.01, 'losses': [0.5, 0.4, 0.3]}
        ],
        'beta_study': [
            {'beta': 0.8, 'losses': [0.5, 0.42, 0.35]},
            {'beta': 0.9, 'losses': [0.5, 0.4, 0.3]}
        ],
        'curvature_study': [
            {'curvature_influence': 0.05, 'losses': [0.5, 0.41, 0.32]},
            {'curvature_influence': 0.1, 'losses': [0.5, 0.4, 0.3]}
        ]
    }
    
    # Set minimal configuration
    config = {
        'seed': 42
    }
    
    # Evaluate synthetic results
    print("\nEvaluating synthetic results...")
    evaluate_synthetic(synthetic_results, config)
    
    # Evaluate CIFAR-10 results
    print("\nEvaluating CIFAR-10 results...")
    evaluate_cifar10(cifar10_results, config)
    
    # Evaluate MNIST results
    print("\nEvaluating MNIST results...")
    evaluate_mnist(mnist_results, config)
    
    # Evaluate ablation study results
    print("\nEvaluating ablation study results...")
    evaluate_ablation_study(ablation_results, config)
    
    print("\nQuick evaluation test completed successfully!")

if __name__ == "__main__":
    # Run quick test to verify code execution
    quick_test()
