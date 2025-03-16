"""
Data preprocessing module for ACM optimizer experiments.

This module handles data loading and preprocessing for the experiments:
1. Synthetic function benchmarking
2. CIFAR-10 image classification
3. Ablation studies
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

def create_synthetic_data(func_name, init_params=None):
    """
    Create synthetic data for optimization benchmarks.
    
    Args:
        func_name (str): Name of the function ('rosenbrock' or 'ill_conditioned')
        init_params (list, optional): Initial parameters for optimization
        
    Returns:
        tuple: (function, initial parameters)
    """
    if func_name == 'rosenbrock':
        # Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        def rosenbrock(params):
            x, y = params[0], params[1]
            return (1 - x)**2 + 100 * (y - x**2)**2
        
        # Default initial parameters if not provided
        if init_params is None:
            init_params = [-1.5, 2.0]
            
        return rosenbrock, init_params
    
    elif func_name == 'ill_conditioned':
        # Ill-conditioned quadratic function: f(x) = 0.5 * x^T A x
        def ill_conditioned_quadratic(params):
            # Define A with eigenvalues 1 and 100 for a simple 2D case
            A = torch.tensor([[1.0, 0.0],
                             [0.0, 100.0]])
            params_vec = params.view(-1, 1)  # Ensure column vector
            return 0.5 * (params_vec.t() @ A @ params_vec)[0,0]
        
        # Default initial parameters if not provided
        if init_params is None:
            init_params = [5.0, -5.0]
            
        return ill_conditioned_quadratic, init_params
    
    else:
        raise ValueError(f"Unknown function name: {func_name}")

def load_cifar10(batch_size=128, num_workers=2, data_dir='./data'):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        data_dir (str): Directory to store the dataset
        
    Returns:
        tuple: (train_loader, val_loader, classes)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, val_loader, classes

def get_subset_loaders(train_loader, val_loader, num_batches=10):
    """
    Create subset data loaders for quick testing.
    
    Args:
        train_loader (DataLoader): Original training data loader
        val_loader (DataLoader): Original validation data loader
        num_batches (int): Number of batches to include in the subset
        
    Returns:
        tuple: (subset_train_loader, subset_val_loader)
    """
    # Extract a few batches for quick testing
    train_subset = []
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_batches:
            break
        train_subset.append((images, labels))
    
    val_subset = []
    for i, (images, labels) in enumerate(val_loader):
        if i >= num_batches:
            break
        val_subset.append((images, labels))
    
    # Create custom batch samplers that yield the cached batches
    class SubsetSampler:
        def __init__(self, subset):
            self.subset = subset
            # Add dataset attribute with length for compatibility
            self.dataset = type('DummyDataset', (), {'__len__': lambda _: sum(batch[0].size(0) for batch in subset)})()
        
        def __iter__(self):
            for batch in self.subset:
                yield batch
        
        def __len__(self):
            return len(self.subset)
    
    # Return the subset loaders
    return SubsetSampler(train_subset), SubsetSampler(val_subset)
