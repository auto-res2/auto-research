"""Utility functions for experiments."""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the device to use (GPU if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_loss_curves(results, title, save_path=None):
    """Plot loss curves for different optimizers.
    
    Args:
        results (dict): Dictionary mapping optimizer names to lists of losses
        title (str): Title of the plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def log_experiment_results(experiment_name, results):
    """Log experiment results to console.
    
    Args:
        experiment_name (str): Name of the experiment
        results (dict): Dictionary containing experiment results
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    
    print(f"{'='*80}\n")


def get_timestamp():
    """Get current timestamp as a string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
