"""
Data preprocessing for ABS-Diff experiments.

This script handles data loading and preprocessing for the 
Adaptive Bayesian SDE-Guided Diffusion experiments.
"""

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def load_cifar10(batch_size=64, num_workers=2):
    """
    Load and preprocess the CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    transform = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        transform=transform, 
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
