"""Data preprocessing for experiments."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config.experiment_config import (
    CIFAR_BATCH_SIZE,
    MNIST_BATCH_SIZE,
    RANDOM_SEED,
)
from src.utils.utils import set_seed


def load_cifar10():
    """Load and preprocess CIFAR-10 dataset.
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    set_seed(RANDOM_SEED)
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CIFAR_BATCH_SIZE,
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CIFAR_BATCH_SIZE,
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader


def load_mnist():
    """Load and preprocess MNIST dataset.
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    set_seed(RANDOM_SEED)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=MNIST_BATCH_SIZE,
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=MNIST_BATCH_SIZE,
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader


def get_synthetic_functions():
    """Get synthetic optimization functions.
    
    Returns:
        dict: Dictionary of synthetic functions
    """
    # Define a convex quadratic function: f(x) = 0.5 * x^T A x - b^T x
    def quadratic_loss(x, A, b):
        return 0.5 * x @ A @ x - b @ x
    
    # Define a modified Rosenbrock-like function (a simple nonconvex function)
    def rosenbrock_loss(x):
        # Here x is assumed to be a 2D tensor
        a = 1.0
        b = 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    return {
        "quadratic": quadratic_loss,
        "rosenbrock": rosenbrock_loss
    }
