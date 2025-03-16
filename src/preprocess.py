"""
Data preprocessing module for ACM optimizer experiments.

This module handles loading and preprocessing of datasets for the experiments
comparing the Adaptive Curvature Momentum (ACM) optimizer with other optimizers.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_cifar10_data(data_dir='./data', batch_size=128, num_workers=4, val_split=0.1):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        data_dir (str): Directory to store the dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        val_split (float): Fraction of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation and test sets
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Split training data into train and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_quadratic_function_data(n_samples=1000, dimension=10, curvature=5.0):
    """
    Generate synthetic data for a quadratic function experiment.
    
    This function creates a simple quadratic function dataset for testing
    optimizer behavior on functions with different curvature properties.
    
    Args:
        n_samples (int): Number of samples to generate
        dimension (int): Dimensionality of the data
        curvature (float): Curvature parameter (higher = more curved)
        
    Returns:
        x_data: Input data points
        y_data: Target values (function outputs)
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate random points
    x_data = torch.randn(n_samples, dimension)
    
    # Compute quadratic function values: f(x) = 0.5 * curvature * ||x||^2
    y_data = 0.5 * curvature * torch.sum(x_data ** 2, dim=1)
    
    return x_data, y_data


def prepare_data(config):
    """
    Prepare data based on configuration.
    
    Args:
        config (dict): Configuration dictionary with data parameters
        
    Returns:
        data: Dictionary containing data loaders or datasets
    """
    experiment_type = config.get('experiment_type', 'cifar10')
    
    if experiment_type == 'cifar10':
        train_loader, val_loader, test_loader = get_cifar10_data(
            data_dir=config.get('data_dir', './data'),
            batch_size=config.get('batch_size', 128),
            num_workers=config.get('num_workers', 4),
            val_split=config.get('val_split', 0.1)
        )
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }
    
    elif experiment_type == 'quadratic':
        x_data, y_data = get_quadratic_function_data(
            n_samples=config.get('n_samples', 1000),
            dimension=config.get('dimension', 10),
            curvature=config.get('curvature', 5.0)
        )
        return {
            'x_data': x_data,
            'y_data': y_data
        }
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
