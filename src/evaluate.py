import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from train import SimpleCNN, MNISTNet

def evaluate_synthetic_results(results, config):
    """
    Evaluate and visualize results from synthetic optimization experiments.
    
    Args:
        results (list): List of dictionaries containing results for each optimizer
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Extract results for each optimizer
    optimizers = [result['optimizer'] for result in results]
    
    # Prepare data for plotting
    quadratic_losses = {opt: result['quadratic']['losses'] for opt, result in zip(optimizers, results)}
    quadratic_final_losses = {opt: result['quadratic']['final_loss'] for opt, result in zip(optimizers, results)}
    quadratic_times = {opt: result['quadratic']['training_time'] for opt, result in zip(optimizers, results)}
    
    # Evaluate Rosenbrock results if available
    if results[0]['rosenbrock']['losses']:
        rosenbrock_losses = {opt: result['rosenbrock']['losses'] for opt, result in zip(optimizers, results)}
        rosenbrock_final_losses = {opt: result['rosenbrock']['final_loss'] for opt, result in zip(optimizers, results)}
        rosenbrock_times = {opt: result['rosenbrock']['training_time'] for opt, result in zip(optimizers, results)}
    else:
        rosenbrock_losses = {}
        rosenbrock_final_losses = {}
        rosenbrock_times = {}
    
    # Print evaluation results
    print("\n=== Synthetic Optimization Benchmark Results ===")
    
    # Quadratic function results
    print("\nQuadratic Function Results:")
    print("-" * 50)
    print(f"{'Optimizer':<10} | {'Final Loss':<15} | {'Training Time (s)':<20}")
    print("-" * 50)
    
    for opt in optimizers:
        print(f"{opt:<10} | {quadratic_final_losses[opt]:<15.6f} | {quadratic_times[opt]:<20.4f}")
    
    # Rosenbrock function results (if available)
    if rosenbrock_losses:
        print("\nRosenbrock Function Results:")
        print("-" * 50)
        print(f"{'Optimizer':<10} | {'Final Loss':<15} | {'Training Time (s)':<20}")
        print("-" * 50)
        
        for opt in optimizers:
            print(f"{opt:<10} | {rosenbrock_final_losses[opt]:<15.6f} | {rosenbrock_times[opt]:<20.4f}")
    
    # Create evaluation metrics
    evaluation = {
        'quadratic': {
            'best_optimizer': min(quadratic_final_losses, key=quadratic_final_losses.get),
            'best_loss': min(quadratic_final_losses.values()),
            'fastest_optimizer': min(quadratic_times, key=quadratic_times.get),
            'fastest_time': min(quadratic_times.values()),
            'final_losses': quadratic_final_losses,
            'training_times': quadratic_times
        }
    }
    
    if rosenbrock_losses:
        evaluation['rosenbrock'] = {
            'best_optimizer': min(rosenbrock_final_losses, key=rosenbrock_final_losses.get),
            'best_loss': min(rosenbrock_final_losses.values()),
            'fastest_optimizer': min(rosenbrock_times, key=rosenbrock_times.get),
            'fastest_time': min(rosenbrock_times.values()),
            'final_losses': rosenbrock_final_losses,
            'training_times': rosenbrock_times
        }
    
    # Generate plots if specified in config
    if config.get('generate_plots', True):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Plot quadratic function convergence
        plt.figure(figsize=(10, 6))
        for opt in optimizers:
            plt.plot(quadratic_losses[opt], label=opt)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Quadratic Function Optimization')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('logs/quadratic_convergence.png')
        
        # Plot Rosenbrock function convergence (if available)
        if rosenbrock_losses:
            plt.figure(figsize=(10, 6))
            for opt in optimizers:
                plt.plot(rosenbrock_losses[opt], label=opt)
            
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Rosenbrock Function Optimization')
            plt.legend()
            plt.yscale('log')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig('logs/rosenbrock_convergence.png')
    
    return evaluation

def evaluate_cifar10_results(results, config):
    """
    Evaluate and visualize results from CIFAR-10 experiments.
    
    Args:
        results (dict): Dictionary containing results for each optimizer
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    optimizers = list(results.keys())
    
    # Extract metrics for comparison
    train_accs = {opt: results[opt]['final_train_acc'] for opt in optimizers}
    test_accs = {opt: results[opt]['final_test_acc'] for opt in optimizers}
    train_losses = {opt: results[opt]['final_train_loss'] for opt in optimizers}
    test_losses = {opt: results[opt]['final_test_loss'] for opt in optimizers}
    training_times = {opt: results[opt]['total_time'] for opt in optimizers}
    
    # Print evaluation results
    print("\n=== CIFAR-10 Experiment Results ===")
    print("-" * 80)
    print(f"{'Optimizer':<10} | {'Train Acc (%)':<15} | {'Test Acc (%)':<15} | {'Train Loss':<15} | {'Test Loss':<15} | {'Time (s)':<10}")
    print("-" * 80)
    
    for opt in optimizers:
        print(f"{opt:<10} | {train_accs[opt]:<15.2f} | {test_accs[opt]:<15.2f} | {train_losses[opt]:<15.6f} | {test_losses[opt]:<15.6f} | {training_times[opt]:<10.2f}")
    
    # Create evaluation metrics
    evaluation = {
        'best_train_acc': {
            'optimizer': max(train_accs, key=train_accs.get),
            'value': max(train_accs.values())
        },
        'best_test_acc': {
            'optimizer': max(test_accs, key=test_accs.get),
            'value': max(test_accs.values())
        },
        'best_train_loss': {
            'optimizer': min(train_losses, key=train_losses.get),
            'value': min(train_losses.values())
        },
        'best_test_loss': {
            'optimizer': min(test_losses, key=test_losses.get),
            'value': min(test_losses.values())
        },
        'fastest_optimizer': {
            'optimizer': min(training_times, key=training_times.get),
            'value': min(training_times.values())
        },
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_times': training_times
    }
    
    # Generate plots if specified in config
    if config.get('generate_plots', True):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Plot training accuracy curves
        plt.figure(figsize=(12, 10))
        
        # Plot training accuracy
        plt.subplot(2, 2, 1)
        for opt in optimizers:
            plt.plot(results[opt]['history']['train_acc'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Plot test accuracy
        plt.subplot(2, 2, 2)
        for opt in optimizers:
            plt.plot(results[opt]['history']['test_acc'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Plot training loss
        plt.subplot(2, 2, 3)
        for opt in optimizers:
            plt.plot(results[opt]['history']['train_loss'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Plot test loss
        plt.subplot(2, 2, 4)
        for opt in optimizers:
            plt.plot(results[opt]['history']['test_loss'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Loss')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('logs/cifar10_results.png')
        
        # Plot epoch times
        plt.figure(figsize=(10, 6))
        for opt in optimizers:
            plt.plot(results[opt]['history']['epoch_times'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Epoch Training Times')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('logs/cifar10_epoch_times.png')
    
    return evaluation

def evaluate_mnist_results(results, config):
    """
    Evaluate and visualize results from MNIST experiments (ablation study).
    
    Args:
        results (dict): Dictionary containing results for each optimizer configuration
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    optimizer_configs = list(results.keys())
    
    # Extract metrics for comparison
    train_accs = {opt: results[opt]['final_train_acc'] for opt in optimizer_configs}
    test_accs = {opt: results[opt]['final_test_acc'] for opt in optimizer_configs}
    train_losses = {opt: results[opt]['final_train_loss'] for opt in optimizer_configs}
    test_losses = {opt: results[opt]['final_test_loss'] for opt in optimizer_configs}
    training_times = {opt: results[opt]['total_time'] for opt in optimizer_configs}
    
    # Print evaluation results
    print("\n=== MNIST Experiment Results (Ablation Study) ===")
    print("-" * 80)
    print(f"{'Optimizer':<20} | {'Train Acc (%)':<15} | {'Test Acc (%)':<15} | {'Train Loss':<15} | {'Test Loss':<15} | {'Time (s)':<10}")
    print("-" * 80)
    
    for opt in optimizer_configs:
        print(f"{opt:<20} | {train_accs[opt]:<15.2f} | {test_accs[opt]:<15.2f} | {train_losses[opt]:<15.6f} | {test_losses[opt]:<15.6f} | {training_times[opt]:<10.2f}")
    
    # Create evaluation metrics
    evaluation = {
        'best_train_acc': {
            'optimizer': max(train_accs, key=train_accs.get),
            'value': max(train_accs.values())
        },
        'best_test_acc': {
            'optimizer': max(test_accs, key=test_accs.get),
            'value': max(test_accs.values())
        },
        'best_train_loss': {
            'optimizer': min(train_losses, key=train_losses.get),
            'value': min(train_losses.values())
        },
        'best_test_loss': {
            'optimizer': min(test_losses, key=test_losses.get),
            'value': min(test_losses.values())
        },
        'fastest_optimizer': {
            'optimizer': min(training_times, key=training_times.get),
            'value': min(training_times.values())
        },
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_times': training_times
    }
    
    # Generate plots if specified in config
    if config.get('generate_plots', True):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Plot training accuracy curves
        plt.figure(figsize=(12, 10))
        
        # Plot training accuracy
        plt.subplot(2, 2, 1)
        for opt in optimizer_configs:
            plt.plot(results[opt]['history']['train_acc'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Plot test accuracy
        plt.subplot(2, 2, 2)
        for opt in optimizer_configs:
            plt.plot(results[opt]['history']['test_acc'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Plot training loss
        plt.subplot(2, 2, 3)
        for opt in optimizer_configs:
            plt.plot(results[opt]['history']['train_loss'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Plot test loss
        plt.subplot(2, 2, 4)
        for opt in optimizer_configs:
            plt.plot(results[opt]['history']['test_loss'], label=opt)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Loss')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('logs/mnist_results.png')
        
        # If ablation study is enabled, create heatmap of hyperparameter performance
        if config.get('run_ablation', True) and len(optimizer_configs) > 3:  # More than just the standard optimizers
            # Extract ACM configurations
            acm_configs = [opt for opt in optimizer_configs if opt.startswith('ACM_b')]
            
            if acm_configs:
                # Extract beta and curvature_influence values
                betas = sorted(list(set([results[opt]['config']['beta'] for opt in acm_configs])))
                cis = sorted(list(set([results[opt]['config']['curvature_influence'] for opt in acm_configs])))
                
                # Create heatmap data for test accuracy
                heatmap_data = np.zeros((len(betas), len(cis)))
                
                for i, beta in enumerate(betas):
                    for j, ci in enumerate(cis):
                        # Find the optimizer with this configuration
                        for opt in acm_configs:
                            if (results[opt]['config']['beta'] == beta and 
                                results[opt]['config']['curvature_influence'] == ci):
                                heatmap_data[i, j] = results[opt]['final_test_acc']
                                break
                
                # Plot heatmap
                plt.figure(figsize=(10, 8))
                plt.imshow(heatmap_data, interpolation='nearest', cmap='viridis')
                plt.colorbar(label='Test Accuracy (%)')
                
                # Set ticks and labels
                plt.xticks(np.arange(len(cis)), [str(ci) for ci in cis])
                plt.yticks(np.arange(len(betas)), [str(beta) for beta in betas])
                
                plt.xlabel('Curvature Influence')
                plt.ylabel('Beta (Momentum)')
                plt.title('ACM Hyperparameter Sensitivity Analysis')
                
                # Add text annotations
                for i in range(len(betas)):
                    for j in range(len(cis)):
                        plt.text(j, i, f"{heatmap_data[i, j]:.1f}",
                                ha="center", va="center", color="w" if heatmap_data[i, j] < 90 else "black")
                
                plt.tight_layout()
                plt.savefig('logs/acm_hyperparameter_heatmap.png')
    
    return evaluation

def load_and_evaluate_model(model_path, test_loader, device=None):
    """
    Load a saved model and evaluate it on the test set.
    
    Args:
        model_path (str): Path to the saved model
        test_loader (DataLoader): Test data loader
        device (torch.device, optional): Device to use for computation
        
    Returns:
        tuple: (model, test_accuracy, test_loss)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine model type from filename
    if 'cifar10' in model_path:
        model = SimpleCNN()
    elif 'mnist' in model_path:
        model = MNISTNet()
    else:
        raise ValueError(f"Cannot determine model type from path: {model_path}")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Evaluate model
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Calculate average loss and accuracy
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return model, test_acc, test_loss

def evaluate_results(results, config):
    """
    Main evaluation function that evaluates results from all experiments.
    
    Args:
        results (dict): Dictionary containing results from all experiments
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing evaluation metrics for all experiments
    """
    evaluation = {}
    
    # Evaluate synthetic results
    if 'synthetic' in results:
        print("\nEvaluating synthetic optimization results...")
        evaluation['synthetic'] = evaluate_synthetic_results(results['synthetic'], config)
    
    # Evaluate CIFAR-10 results
    if 'cifar10' in results:
        print("\nEvaluating CIFAR-10 results...")
        evaluation['cifar10'] = evaluate_cifar10_results(results['cifar10'], config)
    
    # Evaluate MNIST results
    if 'mnist' in results:
        print("\nEvaluating MNIST results...")
        evaluation['mnist'] = evaluate_mnist_results(results['mnist'], config)
    
    # Print overall summary
    print("\n=== Overall Evaluation Summary ===")
    
    if 'synthetic' in evaluation:
        print("\nSynthetic Optimization:")
        print(f"Best optimizer for quadratic function: {evaluation['synthetic']['quadratic']['best_optimizer']} (Loss: {evaluation['synthetic']['quadratic']['best_loss']:.6f})")
        if 'rosenbrock' in evaluation['synthetic']:
            print(f"Best optimizer for Rosenbrock function: {evaluation['synthetic']['rosenbrock']['best_optimizer']} (Loss: {evaluation['synthetic']['rosenbrock']['best_loss']:.6f})")
    
    if 'cifar10' in evaluation:
        print("\nCIFAR-10:")
        print(f"Best test accuracy: {evaluation['cifar10']['best_test_acc']['optimizer']} ({evaluation['cifar10']['best_test_acc']['value']:.2f}%)")
        print(f"Fastest optimizer: {evaluation['cifar10']['fastest_optimizer']['optimizer']} ({evaluation['cifar10']['fastest_optimizer']['value']:.2f}s)")
    
    if 'mnist' in evaluation:
        print("\nMNIST (Ablation Study):")
        print(f"Best test accuracy: {evaluation['mnist']['best_test_acc']['optimizer']} ({evaluation['mnist']['best_test_acc']['value']:.2f}%)")
        print(f"Fastest optimizer: {evaluation['mnist']['fastest_optimizer']['optimizer']} ({evaluation['mnist']['fastest_optimizer']['value']:.2f}s)")
    
    return evaluation

if __name__ == "__main__":
    # Simple test to verify the evaluation works
    import torch
    import torch.nn as nn
    
    # Create a simple model and data
    model = nn.Linear(10, 1)
    
    # Create synthetic results for testing
    synthetic_results = [
        {
            'optimizer': 'ACM',
            'quadratic': {
                'losses': [10, 5, 2, 1, 0.5],
                'final_loss': 0.5,
                'training_time': 0.1
            },
            'rosenbrock': {
                'losses': [100, 50, 20, 10, 5],
                'final_loss': 5,
                'training_time': 0.2
            }
        },
        {
            'optimizer': 'Adam',
            'quadratic': {
                'losses': [10, 6, 3, 2, 1],
                'final_loss': 1,
                'training_time': 0.15
            },
            'rosenbrock': {
                'losses': [100, 60, 30, 20, 10],
                'final_loss': 10,
                'training_time': 0.25
            }
        }
    ]
    
    # Test evaluation
    config = {'generate_plots': False}
    evaluation = evaluate_synthetic_results(synthetic_results, config)
    print("Evaluation test completed successfully.")
