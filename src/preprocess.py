"""
Data preprocessing module for the auto-research project.
Contains functions for loading and preprocessing datasets for the experiments.
"""

import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set seeds for reproducibility
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get the available device (GPU or CPU).
    
    Returns:
        torch.device: The device to use for tensor operations
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cifar10(batch_size=128, num_workers=2):
    """
    Load and preprocess the CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker threads for data loading
        
    Returns:
        tuple: (trainloader, testloader, classes) containing data loaders and class names
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
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def generate_two_moons_data(n_samples=1000, noise=0.2, test_size=0.2, seed=42):
    """
    Generate the two-moons synthetic dataset.
    
    Args:
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add to the data
        test_size (float): Proportion of data to use for testing
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor) containing
               training and validation data as PyTorch tensors
    """
    # Generate two-moons dataset
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

def synthetic_function(xy, a=1.0, b=2.0, c=-1.0):
    """
    A modified quadratic function with saddle regions for optimization experiments.
    
    Args:
        xy (torch.Tensor): Input tensor of shape (2,)
        a (float): Coefficient for x^2 term
        b (float): Coefficient for y^2 term
        c (float): Coefficient for x*y term
        
    Returns:
        torch.Tensor: Function value at the given point
    """
    x, y = xy[0], xy[1]
    return 0.5 * (a * x**2 + b * y**2) + c * x * y
