import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generate synthetic data for optimization benchmarks.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing synthetic datasets
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate data for quadratic function
    # f(x) = 0.5 * x^T A x - b^T x
    A = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
    b = torch.tensor([1.0, 1.0])
    
    # Generate random starting points for optimization
    quadratic_data = torch.randn(n_samples, 2)
    
    # Generate data for Rosenbrock function
    # f(x) = (a - x[0])^2 + b * (x[1] - x[0]^2)^2
    rosenbrock_data = torch.randn(n_samples, 2)
    
    return {
        'quadratic': {
            'data': quadratic_data,
            'A': A,
            'b': b
        },
        'rosenbrock': {
            'data': rosenbrock_data,
            'a': 1.0,
            'b': 100.0
        }
    }

def load_cifar10(batch_size=128, download=True):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loaders
        download: Whether to download the dataset if not available
        
    Returns:
        Dictionary containing train and test data loaders
    """
    # Define transformations for CIFAR-10
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
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=download, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=download, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return {
        'train_loader': trainloader,
        'test_loader': testloader,
        'classes': ('plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')
    }

def load_mnist(batch_size=128, download=True):
    """
    Load and preprocess MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        download: Whether to download the dataset if not available
        
    Returns:
        Dictionary containing train and test data loaders
    """
    # Define transformations for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=download, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=download, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return {
        'train_loader': trainloader,
        'test_loader': testloader
    }

def preprocess_data(config=None):
    """
    Main function to preprocess data for all experiments.
    
    Args:
        config: Configuration dictionary with preprocessing parameters
        
    Returns:
        Dictionary containing all preprocessed datasets
    """
    # Create necessary directories
    create_directories()
    
    # Set default configuration if not provided
    if config is None:
        config = {
            'synthetic': {
                'n_samples': 1000,
                'seed': 42
            },
            'cifar10': {
                'batch_size': 128,
                'download': True
            },
            'mnist': {
                'batch_size': 128,
                'download': True
            }
        }
    
    # Generate/load datasets
    synthetic_data = generate_synthetic_data(
        n_samples=config['synthetic']['n_samples'],
        seed=config['synthetic']['seed']
    )
    
    cifar10_data = load_cifar10(
        batch_size=config['cifar10']['batch_size'],
        download=config['cifar10']['download']
    )
    
    mnist_data = load_mnist(
        batch_size=config['mnist']['batch_size'],
        download=config['mnist']['download']
    )
    
    return {
        'synthetic': synthetic_data,
        'cifar10': cifar10_data,
        'mnist': mnist_data
    }

if __name__ == "__main__":
    # Test the preprocessing functions
    data = preprocess_data()
    print("Synthetic data shapes:")
    print(f"Quadratic data shape: {data['synthetic']['quadratic']['data'].shape}")
    print(f"Rosenbrock data shape: {data['synthetic']['rosenbrock']['data'].shape}")
    
    print("\nCIFAR-10 dataset:")
    print(f"Number of training batches: {len(data['cifar10']['train_loader'])}")
    print(f"Number of test batches: {len(data['cifar10']['test_loader'])}")
    
    print("\nMNIST dataset:")
    print(f"Number of training batches: {len(data['mnist']['train_loader'])}")
    print(f"Number of test batches: {len(data['mnist']['test_loader'])}")
