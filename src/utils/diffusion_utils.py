"""
Utility functions for diffusion models and CGCD implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

def seed_everything(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def get_device():
    """
    Get the device to use (CUDA if available, else CPU).
    
    Returns:
        torch.device: The device to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available. Using CPU.")
    return device

def save_model(model, path, filename):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        path: Directory to save to
        filename: Name of checkpoint file
    """
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Model saved to {os.path.join(path, filename)}")

def load_model(model, path, filename):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        path: Directory to load from
        filename: Name of checkpoint file
        
    Returns:
        model: Loaded model
    """
    model.load_state_dict(torch.load(os.path.join(path, filename)))
    print(f"Model loaded from {os.path.join(path, filename)}")
    return model

def create_tensorboard_writer(log_dir):
    """
    Create TensorBoard writer.
    
    Args:
        log_dir: Directory for TensorBoard logs
        
    Returns:
        SummaryWriter object
    """
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)

def plot_images(images, title, save_path=None):
    """
    Plot a grid of images.
    
    Args:
        images: Tensor of images [B, C, H, W]
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Denormalize if needed
    if images.min() < 0:
        images = (images + 1) / 2
    
    # Convert to numpy and move channels to last dimension
    images = images.detach().cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Plot
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < images.shape[0]:
            if images.shape[3] == 1:
                ax.imshow(images[i, :, :, 0], cmap='gray')
            else:
                ax.imshow(images[i])
            ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def get_condition(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoding.
    
    Args:
        labels: Tensor of integer labels
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        One-hot encoded tensor
    """
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
