"""
Utility functions for running experiments with the ACM optimizer.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import time
import json
from datetime import datetime


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the device to use for training (GPU if available, otherwise CPU).
    
    Returns:
        device: PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    return device


def save_model(model, optimizer, epoch, loss, accuracy, path):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        accuracy (float): Current accuracy
        path (str): Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, optimizer, path):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        path (str): Path to the checkpoint
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        epoch (int): Epoch from the checkpoint
        loss (float): Loss from the checkpoint
        accuracy (float): Accuracy from the checkpoint
    """
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    
    return model, optimizer, epoch, loss, accuracy


def save_config(config, path):
    """
    Save experiment configuration to a JSON file.
    
    Args:
        config (dict): Configuration dictionary
        path (str): Path to save the configuration
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {path}")


def load_config(path):
    """
    Load experiment configuration from a JSON file.
    
    Args:
        path (str): Path to the configuration file
        
    Returns:
        config (dict): Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    
    return config


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, title, path):
    """
    Plot learning curves for training and validation.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        train_accuracies (list): List of training accuracies
        val_accuracies (list): List of validation accuracies
        title (str): Plot title
        path (str): Path to save the plot
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    print(f"Learning curves saved to {path}")


def plot_optimizer_comparison(results, title, path):
    """
    Plot comparison of different optimizers.
    
    Args:
        results (dict): Dictionary containing results for different optimizers
        title (str): Plot title
        path (str): Path to save the plot
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot function values
    for opt_name, data in results.items():
        axes[0].plot(data["f_vals"], label=opt_name)
    
    axes[0].set_title(f'{title}: Convergence Curve')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Function Value')
    axes[0].legend()
    
    # Plot learning rates
    for opt_name, data in results.items():
        axes[1].plot(data["lr_trace"], label=opt_name)
    
    axes[1].set_title('Adaptive Learning Rates')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Learning Rate')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    print(f"Optimizer comparison saved to {path}")


class ExperimentLogger:
    """
    Logger for experiment results.
    
    Args:
        log_dir (str): Directory to save logs
        experiment_name (str): Name of the experiment
    """
    
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"=== {experiment_name} ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log(self, message):
        """
        Log a message.
        
        Args:
            message (str): Message to log
        """
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
    
    def log_metrics(self, epoch, metrics):
        """
        Log metrics for an epoch.
        
        Args:
            epoch (int): Current epoch
            metrics (dict): Dictionary of metrics
        """
        message = f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.log(message)
    
    def log_config(self, config):
        """
        Log experiment configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.log("=== Configuration ===")
        for k, v in config.items():
            self.log(f"{k}: {v}")
        self.log("")
    
    def log_summary(self, metrics):
        """
        Log summary of the experiment.
        
        Args:
            metrics (dict): Dictionary of metrics
        """
        self.log("\n=== Summary ===")
        for k, v in metrics.items():
            self.log(f"{k}: {v}")
        self.log(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
