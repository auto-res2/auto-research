import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def create_data_directories():
    """Create necessary data directories if they don't exist."""
    os.makedirs('data/cifar10', exist_ok=True)
    os.makedirs('data/mnist', exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)

def load_cifar10(batch_size=128):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader) containing the data loaders
    """
    # Define transformations for the training and test sets
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
    
    # Load the CIFAR-10 training and test datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', 
        train=False,
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader

def load_mnist(batch_size=128):
    """
    Load and preprocess MNIST dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader) containing the data loaders
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the MNIST training and test datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data/mnist', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data/mnist', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader

def generate_synthetic_data():
    """
    Generate synthetic data for optimization benchmarks.
    
    Returns:
        dict: Dictionary containing synthetic optimization problems
    """
    # Create a positive definite matrix for quadratic function
    A = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
    b = torch.tensor([1.0, 1.0])
    
    # Save synthetic data
    synthetic_data = {
        'quadratic': {
            'A': A,
            'b': b
        },
        'rosenbrock': {
            'a': 1.0,
            'b': 100.0
        }
    }
    
    # Save to file
    torch.save(synthetic_data, 'data/synthetic/optimization_problems.pt')
    
    return synthetic_data

def preprocess_data(config=None):
    """
    Main function to preprocess all datasets.
    
    Args:
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Dictionary containing all preprocessed data
    """
    # Create necessary directories
    create_data_directories()
    
    # Set default batch size if not provided in config
    batch_size = 128
    if config and 'batch_size' in config:
        batch_size = config['batch_size']
    
    # Load datasets
    cifar10_loaders = load_cifar10(batch_size)
    mnist_loaders = load_mnist(batch_size)
    synthetic_data = generate_synthetic_data()
    
    # Return all data
    return {
        'cifar10': cifar10_loaders,
        'mnist': mnist_loaders,
        'synthetic': synthetic_data
    }

if __name__ == "__main__":
    # When run directly, preprocess all data
    preprocess_data()
    print("Data preprocessing completed successfully.")
