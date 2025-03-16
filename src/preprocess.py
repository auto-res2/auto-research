#!/usr/bin/env python3
"""
Data preprocessing module for ACM optimizer experiments.

This module handles loading and preprocessing the CIFAR-10 and CIFAR-100 datasets
for the experiments comparing different optimizers.
"""

import os
import yaml
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_config(config_path='config/acm_experiments.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(config=None):
    """
    Get the device to use for training.
    
    Args:
        config (dict, optional): Configuration dictionary.
        
    Returns:
        torch.device: Device to use for training.
    """
    if config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['general']['device'] if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    return device


def get_cifar10_dataloaders(batch_size=128, data_dir='data', test_run=False):
    """
    Get CIFAR-10 dataloaders for training and testing.
    
    Args:
        batch_size (int): Batch size for training and testing.
        data_dir (str): Directory to store the dataset.
        test_run (bool): Whether this is a test run with reduced dataset size.
        
    Returns:
        tuple: (train_loader, test_loader) for CIFAR-10.
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Data transforms for training and testing
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
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # If test run, use a subset of the data
    if test_run:
        # Use 10% of the training data for test runs
        train_size = len(trainset) // 10
        test_size = len(testset) // 10
        indices = torch.randperm(len(trainset))[:train_size]
        trainset = torch.utils.data.Subset(trainset, indices)
        indices = torch.randperm(len(testset))[:test_size]
        testset = torch.utils.data.Subset(testset, indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"CIFAR-10 dataset loaded. Training samples: {len(trainset)}, Test samples: {len(testset)}")
    
    return train_loader, test_loader


def get_cifar100_dataloaders(batch_size=128, data_dir='data', test_run=False):
    """
    Get CIFAR-100 dataloaders for training and testing.
    
    Args:
        batch_size (int): Batch size for training and testing.
        data_dir (str): Directory to store the dataset.
        test_run (bool): Whether this is a test run with reduced dataset size.
        
    Returns:
        tuple: (train_loader, test_loader) for CIFAR-100.
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Data transforms for training and testing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # If test run, use a subset of the data
    if test_run:
        # Use 10% of the training data for test runs
        train_size = len(trainset) // 10
        test_size = len(testset) // 10
        indices = torch.randperm(len(trainset))[:train_size]
        trainset = torch.utils.data.Subset(trainset, indices)
        indices = torch.randperm(len(testset))[:test_size]
        testset = torch.utils.data.Subset(testset, indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"CIFAR-100 dataset loaded. Training samples: {len(trainset)}, Test samples: {len(testset)}")
    
    return train_loader, test_loader


def get_dataloaders(config, dataset_name, test_run=False):
    """
    Get dataloaders for the specified dataset.
    
    Args:
        config (dict): Configuration dictionary.
        dataset_name (str): Name of the dataset ('cifar10' or 'cifar100').
        test_run (bool): Whether this is a test run with reduced dataset size.
        
    Returns:
        tuple: (train_loader, test_loader) for the specified dataset.
    """
    data_dir = config['general']['data_dir']
    
    if dataset_name.lower() == 'cifar10':
        if test_run:
            batch_size = config['test_run']['experiment1']['batch_size']
        else:
            batch_size = config['experiment1']['batch_size']
        return get_cifar10_dataloaders(batch_size, data_dir, test_run)
    
    elif dataset_name.lower() == 'cifar100':
        if test_run:
            batch_size = config['test_run']['experiment3']['batch_size']
        else:
            batch_size = config['experiment3']['batch_size']
        return get_cifar100_dataloaders(batch_size, data_dir, test_run)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Get device
    device = get_device(config)
    
    # Test CIFAR-10 dataloaders
    train_loader, test_loader = get_dataloaders(config, 'cifar10', test_run=True)
    
    # Print sample batch
    images, labels = next(iter(train_loader))
    print(f"CIFAR-10 batch shape: {images.shape}")
    print(f"CIFAR-10 labels shape: {labels.shape}")
    
    # Test CIFAR-100 dataloaders
    train_loader, test_loader = get_dataloaders(config, 'cifar100', test_run=True)
    
    # Print sample batch
    images, labels = next(iter(train_loader))
    print(f"CIFAR-100 batch shape: {images.shape}")
    print(f"CIFAR-100 labels shape: {labels.shape}")
