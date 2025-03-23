"""Data preprocessing module for the ACM optimizer experiment."""

import torch
import torchvision
import torchvision.transforms as transforms


def load_cifar10(batch_size=128, num_workers=2):
    """Load CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader) containing the data loaders
    """
    # Define data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load the datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def load_mnist(batch_size=128, num_workers=2):
    """Load MNIST dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader) containing the data loaders
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def create_synthetic_functions():
    """Create synthetic optimization functions.
    
    Returns:
        tuple: (quadratic_loss, rosenbrock_loss) containing the loss functions
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
    
    return quadratic_loss, rosenbrock_loss
