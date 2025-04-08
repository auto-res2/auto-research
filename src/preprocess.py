"""
Preprocessing script for DEALWGAN experiments.
"""

import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataset(batch_size=64, num_workers=4):
    """
    Load and preprocess the CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_dataset(name, batch_size=64, num_workers=4):
    """
    Get dataset by name.
    
    Args:
        name: Dataset name (currently only supports 'cifar10')
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    if name.lower() == 'cifar10':
        return get_cifar10_dataset(batch_size, num_workers)
    else:
        raise ValueError(f"Dataset '{name}' not supported")
