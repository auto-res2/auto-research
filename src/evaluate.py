import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_loader, device):
    """
    Evaluate a model on the test dataset.
    
    Args:
        model: neural network model
        test_loader: DataLoader for test data
        device: device to evaluate on (cuda or cpu)
        
    Returns:
        accuracy: classification accuracy on test set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = correct / total
    return accuracy

def plot_training_curves(results_dict, title="Training Curves", save_path=None):
    """
    Plot training loss and test accuracy curves for different optimizers.
    
    Args:
        results_dict: dictionary with results for different optimizers
        title: plot title
        save_path: where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss
    ax1.set_title(f"{title} - Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    # Test accuracy
    ax2.set_title(f"{title} - Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    
    for optimizer_name, results in results_dict.items():
        ax1.plot(results['train_loss'], label=optimizer_name)
        ax2.plot(results['test_accuracy'], label=optimizer_name)
        
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_synthetic_optimization(results_dict, save_path=None):
    """
    Plot convergence of different optimizers on synthetic functions.
    
    Args:
        results_dict: dictionary with results for different optimizers
        save_path: where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Quadratic function
    ax1.set_title("Quadratic Function Optimization")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    
    # Rosenbrock function
    ax2.set_title("Rosenbrock Function Optimization")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    
    for optimizer_name, results in results_dict.items():
        ax1.plot(results['quadratic'], label=optimizer_name)
        ax2.plot(results['rosenbrock'], label=optimizer_name)
        
    ax1.legend()
    ax2.legend()
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def ablation_study_plot(results_dict, param_name, save_path=None):
    """
    Plot ablation study results for a specific parameter.
    
    Args:
        results_dict: dictionary with results for different parameter values
        param_name: name of the parameter being studied
        save_path: where to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    param_values = []
    accuracies = []
    
    for param_value, results in sorted(results_dict.items()):
        param_values.append(param_value)
        accuracies.append(results['final_accuracy'])
        
    plt.plot(param_values, accuracies, 'o-')
    plt.xlabel(param_name)
    plt.ylabel("Final Test Accuracy")
    plt.title(f"Ablation Study: Effect of {param_name} on Performance")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf()
