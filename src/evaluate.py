"""Evaluation module for the ACM optimizer experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def evaluate_model(model, test_loader, device):
    """Evaluate a model on the test dataset.
    
    Args:
        model (nn.Module): PyTorch model to evaluate
        test_loader (DataLoader): DataLoader for test data
        device (torch.device): Device to evaluate on
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    
    print(f"Test set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{total} ({test_accuracy:.2f}%)")
    
    return test_loss, test_accuracy


def plot_training_curves(results, title, save_path=None):
    """Plot training curves for different optimizers.
    
    Args:
        results (dict): Dictionary with optimizer names as keys and training results as values
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    for opt_name, opt_results in results.items():
        plt.plot(opt_results['losses'], label=opt_name)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    
    # Plot accuracies (if available)
    if 'accuracies' in next(iter(results.values())):
        plt.subplot(1, 2, 2)
        for opt_name, opt_results in results.items():
            plt.plot(opt_results['accuracies'], label=opt_name)
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{title} - Accuracy")
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_synthetic_trajectories(results, title, save_path=None):
    """Plot optimization trajectories for synthetic problems.
    
    Args:
        results (dict): Dictionary with optimizer names as keys and experiment results as values
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    for opt_name, opt_results in results.items():
        plt.plot(opt_results['losses'], label=opt_name)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    
    # Plot trajectories
    plt.subplot(1, 2, 2)
    for opt_name, opt_results in results.items():
        trajectories = opt_results['trajectories']
        plt.plot(trajectories[:, 0], trajectories[:, 1], 'o-', label=opt_name, alpha=0.7)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"{title} - Parameter Trajectories")
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
