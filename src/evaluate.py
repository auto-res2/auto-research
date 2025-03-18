import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import itertools

def evaluate_cifar10_model(test_loader, model, device=None):
    """
    Evaluate a model on CIFAR-10 test set
    
    Args:
        test_loader: DataLoader for test data
        model: Trained model
        device: Device to use for evaluation (None for auto-detection)
        
    Returns:
        float: Accuracy on test set
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    return accuracy

def plot_cifar10_results(histories, save_path='logs/cifar10_results.png'):
    """
    Plot training loss and test accuracy for CIFAR-10 experiments
    
    Args:
        histories (dict): Dictionary mapping optimizer names to training histories
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for optimizer_name, history in histories.items():
        plt.plot(history['train_loss'], label=optimizer_name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss on CIFAR-10')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    for optimizer_name, history in histories.items():
        plt.plot(history['test_acc'], label=optimizer_name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy on CIFAR-10')
    plt.legend()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_rosenbrock_results(trajectories, loss_values, 
                           save_path='logs/rosenbrock_results.png'):
    """
    Plot optimization trajectories and loss values for Rosenbrock function
    
    Args:
        trajectories (dict): Dictionary mapping optimizer names to trajectories
        loss_values (dict): Dictionary mapping optimizer names to loss values
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot trajectories on contour plot
    plt.subplot(2, 1, 1)
    
    # Create contour grid for Rosenbrock function
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    # Plot contour
    plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='jet')
    
    # Plot trajectories
    for optimizer_name, trajectory in trajectories.items():
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', 
                label=optimizer_name, markersize=3)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Trajectories on Rosenbrock Function')
    plt.legend()
    plt.grid(True)
    
    # Plot loss values
    plt.subplot(2, 1, 2)
    for optimizer_name, values in loss_values.items():
        plt.semilogy(values, label=optimizer_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Evolution on Rosenbrock Function')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_text_classification_results(histories, 
                                   save_path='logs/text_classification_results.png'):
    """
    Plot training loss and test accuracy for text classification experiments
    
    Args:
        histories (dict): Dictionary mapping optimizer names to training histories
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for optimizer_name, history in histories.items():
        plt.plot(history['train_loss'], label=optimizer_name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss on Text Classification')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    for optimizer_name, history in histories.items():
        plt.plot(history['test_acc'], label=optimizer_name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy on Text Classification')
    plt.legend()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_hyperparameter_sensitivity(param_histories, 
                                  save_path='logs/hyperparameter_sensitivity.png'):
    """
    Plot hyperparameter sensitivity analysis results
    
    Args:
        param_histories (dict): Dictionary mapping (optimizer, param_value) to histories
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Group by optimizer and parameter type
    optimizers = set()
    param_types = set()
    
    for (optimizer, param_type, param_value), _ in param_histories.items():
        optimizers.add(optimizer)
        param_types.add(param_type)
    
    # Create subplots for each optimizer and parameter type
    for i, optimizer in enumerate(sorted(optimizers)):
        for j, param_type in enumerate(sorted(param_types)):
            plt.subplot(len(optimizers), len(param_types), 
                       i * len(param_types) + j + 1)
            
            # Plot histories for this optimizer and parameter type
            for (opt, p_type, p_value), history in param_histories.items():
                if opt == optimizer and p_type == param_type:
                    plt.plot(history['train_loss'], 
                            label=f"{p_type}={p_value}")
            
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title(f"{optimizer} - {param_type} Sensitivity")
            plt.legend()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_experiment_results(results, filename='logs/experiment_results.json'):
    """
    Save experiment results to a JSON file
    
    Args:
        results (dict): Dictionary containing experiment results
        filename (str): Path to save the results
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = save_experiment_results(value, "")
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            serializable_results[key] = [arr.tolist() for arr in value]
        else:
            serializable_results[key] = value
    
    # Save to file if filename is provided
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    return serializable_results
