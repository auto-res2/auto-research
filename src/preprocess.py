#!/usr/bin/env python3
"""
Data preprocessing module for CGCD experiments.

This module handles:
1. Loading and preprocessing CIFAR10 data for discrete conditioning experiments
2. Generating synthetic data for continuous conditioning experiments
3. Data augmentation and normalization
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

def get_cifar10_dataloader(batch_size=64, num_workers=2, data_dir='./data'):
    """
    Prepare CIFAR10 dataset with normalization.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of workers for data loading
        data_dir: Directory to store dataset
        
    Returns:
        DataLoader for CIFAR10
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"CIFAR10 dataset loaded with {len(trainset)} training samples and {len(testset)} test samples")
    return trainloader, testloader

def generate_continuous_dataset(n_samples=1000, img_size=(3, 32, 32), seed=42):
    """
    Generate synthetic data with continuous conditioning signals.
    
    Args:
        n_samples: Number of samples to generate
        img_size: Size of generated images (channels, height, width)
        seed: Random seed for reproducibility
        
    Returns:
        data: Tensor of synthetic images
        signals: Tensor of continuous conditioning signals
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data = []
    signals = []
    
    for _ in range(n_samples):
        # Create a random image (as a stand-in for a gradient image)
        img = torch.randn(img_size)
        # Continuous signal: random scalar between 0 and 1
        signal = torch.rand(1)
        data.append(img)
        signals.append(signal)
        
    data = torch.stack(data)
    signals = torch.stack(signals).squeeze()
    
    print(f"Generated synthetic dataset with {n_samples} samples")
    return data, signals

def get_continuous_dataloader(batch_size=64, n_samples=500, seed=42):
    """
    Create DataLoader for continuous conditioning experiments.
    
    Args:
        batch_size: Number of samples per batch
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataLoader for synthetic data
    """
    data, signals = generate_continuous_dataset(n_samples=n_samples, seed=seed)
    dataset = TensorDataset(data, signals)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Helper function to get condition as one-hot from labels (for discrete conditioning)
def get_condition(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoding.
    
    Args:
        labels: Tensor of integer labels
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        One-hot encoded tensor
    """
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
