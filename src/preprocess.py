import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def load_mnist(batch_size=64, test_batch_size=1000):
    """
    Load MNIST dataset for the ablation study and hyperparameter sensitivity analysis.
    
    Args:
        batch_size (int): Batch size for training data
        test_batch_size (int): Batch size for test data
        
    Returns:
        tuple: (train_loader, test_loader) containing the DataLoader objects
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def load_cifar10(batch_size=128, test_batch_size=100):
    """
    Load CIFAR-10 dataset for the deep neural network training experiment.
    
    Args:
        batch_size (int): Batch size for training data
        test_batch_size (int): Batch size for test data
        
    Returns:
        tuple: (train_loader, test_loader) containing the DataLoader objects
    """
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def generate_synthetic_data():
    """
    Generate synthetic data for optimization benchmarks.
    
    Returns:
        dict: Dictionary containing synthetic optimization problems
    """
    # For quadratic function: f(x) = 0.5 * x^T A x - b^T x
    A_quadratic = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
    b_quadratic = torch.tensor([1.0, 1.0])
    
    # For Rosenbrock function
    a_rosenbrock = 1.0
    b_rosenbrock = 100.0
    
    return {
        'quadratic': {
            'A': A_quadratic,
            'b': b_quadratic
        },
        'rosenbrock': {
            'a': a_rosenbrock,
            'b': b_rosenbrock
        }
    }

def preprocess_data(config):
    """
    Main preprocessing function that prepares data for all experiments.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing all preprocessed data
    """
    create_directories()
    
    data = {}
    
    # Prepare data for each experiment based on config
    if config.get('run_synthetic', True):
        data['synthetic'] = generate_synthetic_data()
    
    if config.get('run_cifar10', True):
        data['cifar10'] = load_cifar10(
            batch_size=config.get('cifar10_batch_size', 128),
            test_batch_size=config.get('cifar10_test_batch_size', 100)
        )
    
    if config.get('run_mnist', True):
        data['mnist'] = load_mnist(
            batch_size=config.get('mnist_batch_size', 64),
            test_batch_size=config.get('mnist_test_batch_size', 1000)
        )
    
    return data

if __name__ == "__main__":
    # Simple test to verify the preprocessing works
    test_config = {
        'run_synthetic': True,
        'run_cifar10': False,  # Set to False for quick testing
        'run_mnist': False,    # Set to False for quick testing
    }
    
    data = preprocess_data(test_config)
    print("Synthetic data generated successfully:")
    print(f"Quadratic function parameters: A={data['synthetic']['quadratic']['A']}, b={data['synthetic']['quadratic']['b']}")
    print(f"Rosenbrock function parameters: a={data['synthetic']['rosenbrock']['a']}, b={data['synthetic']['rosenbrock']['b']}")
