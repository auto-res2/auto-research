"""
Data preprocessing module for ACM optimizer experiments.

This module handles data loading and preprocessing for the experiments:
1. CIFAR-10 dataset for ResNet-18 and CNN experiments
2. Synthetic functions for optimization trajectory experiments
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_cifar10(test_run=False):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        test_run (bool): If True, use a small subset of data for quick testing
    
    Returns:
        tuple: (train_loader, test_loader) - PyTorch data loaders for training and testing
    """
    # Define transformations for training data (with augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Define transformations for test data (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # If in test mode, use a subset of the data
    if test_run:
        train_samples = 500
        test_samples = 200
        
        # Limit dataset size for testing
        trainset.data = trainset.data[:train_samples]
        trainset.targets = trainset.targets[:train_samples]
        testset.data = testset.data[:test_samples]
        testset.targets = testset.targets[:test_samples]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"CIFAR-10 dataset loaded successfully.")
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    
    return train_loader, test_loader

def prepare_synthetic_functions():
    """
    Prepare synthetic optimization functions for the synthetic landscape experiment.
    
    Returns:
        tuple: (quadratic_fn, rosenbrock_fn) - Functions for optimization
    """
    # Define quadratic function
    def quadratic_fn(x, A=None):
        """
        Quadratic function f(x) = 0.5 * (x^T A x)
        
        Args:
            x (torch.Tensor): Input tensor
            A (torch.Tensor, optional): Positive definite matrix
        
        Returns:
            torch.Tensor: Function value
        """
        if A is None:
            # Default matrix if not provided
            A = torch.tensor([[3.0, 0.5], [0.5, 1.0]], device=x.device)
        return 0.5 * (x @ A @ x)
    
    # Define Rosenbrock function
    def rosenbrock_fn(x, a=1.0, b=100.0):
        """
        Rosenbrock function f(x,y) = (a-x)^2 + b(y-x^2)^2
        
        Args:
            x (torch.Tensor): 2D input tensor [x, y]
            a (float): Parameter a
            b (float): Parameter b
        
        Returns:
            torch.Tensor: Function value
        """
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    print("Synthetic functions prepared successfully.")
    return quadratic_fn, rosenbrock_fn

def get_initial_points():
    """
    Get initial points for synthetic function optimization.
    
    Returns:
        tuple: (x_init_quad, x_init_rosen) - Initial points for quadratic and Rosenbrock functions
    """
    # Initial point for quadratic function
    x_init_quad = torch.tensor([3.0, 2.0], requires_grad=True)
    
    # Initial point for Rosenbrock function
    x_init_rosen = torch.tensor([-1.5, 2.0], requires_grad=True)
    
    return x_init_quad, x_init_rosen

if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader = load_cifar10(test_run=True)
    
    # Test batch retrieval
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test synthetic functions
    quadratic_fn, rosenbrock_fn = prepare_synthetic_functions()
    x_init_quad, x_init_rosen = get_initial_points()
    
    # Test function evaluation
    quad_value = quadratic_fn(x_init_quad)
    rosen_value = rosenbrock_fn(x_init_rosen)
    
    print(f"Quadratic function value at initial point: {quad_value.item()}")
    print(f"Rosenbrock function value at initial point: {rosen_value.item()}")
