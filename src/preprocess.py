"""
Preprocessing module for MCAD experiments.
Handles data loading, transformation, and dataset preparation.
"""
import os
import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def add_noise(x, noise_std=0.1):
    """Add Gaussian noise to input tensor."""
    noise = torch.randn_like(x) * noise_std
    return x + noise

def load_dataset(config):
    """Load dataset based on configuration."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Create data directory if it doesn't exist
    os.makedirs(config['dataset']['data_dir'], exist_ok=True)
    
    if config['dataset']['name'] == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            config['dataset']['data_dir'], 
            train=True, 
            transform=transform, 
            download=True
        )
        test_dataset = datasets.CIFAR10(
            config['dataset']['data_dir'], 
            train=False, 
            transform=transform, 
            download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']['name']}")
        
    return train_dataset, test_dataset

def create_dataloaders(config, train_dataset, test_dataset):
    """Create data loaders for training and testing."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'], 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['dataset']['test_batch_size'], 
        shuffle=False
    )
    
    return train_loader, test_loader

def subsample_dataset(dataset, percentage=0.1):
    """Create a subset of the dataset with the given percentage."""
    total = len(dataset)
    num_samples = max(1, int(total * percentage))
    indices = torch.randperm(total)[:num_samples]
    return Subset(dataset, indices)
