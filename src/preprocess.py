import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_cifar10(batch_size=128, num_workers=2):
    """
    Load and preprocess the CIFAR-10 dataset.
    
    Args:
        batch_size (int): batch size for data loaders
        num_workers (int): number of worker threads for data loading
        
    Returns:
        train_loader, test_loader: PyTorch data loaders for training and testing
    """
    # Data transformations
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
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True, transform=transform_train)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=transform_test)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def load_mnist(batch_size=128, num_workers=2):
    """
    Load and preprocess the MNIST dataset.
    
    Args:
        batch_size (int): batch size for data loaders
        num_workers (int): number of worker threads for data loading
        
    Returns:
        train_loader, test_loader: PyTorch data loaders for training and testing
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    trainset = torchvision.datasets.MNIST(
        root='./data/mnist', train=True, download=True, transform=transform)
    
    testset = torchvision.datasets.MNIST(
        root='./data/mnist', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def create_synthetic_functions():
    """
    Create synthetic optimization test functions.
    
    Returns:
        quadratic_func: a simple quadratic function
        rosenbrock_func: a Rosenbrock-like function
    """
    def quadratic_loss(x, A, b):
        """Quadratic function: f(x) = 0.5 * x^T A x - b^T x"""
        return 0.5 * x @ A @ x - b @ x
        
    def rosenbrock_loss(x):
        """Rosenbrock-like function: f(x) = (a - x[0])^2 + b * (x[1] - x[0]^2)^2"""
        a = 1.0
        b = 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
        
    return quadratic_loss, rosenbrock_loss
