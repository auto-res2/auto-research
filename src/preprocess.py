"""
Preprocessing module for Score-Aligned Step Distillation experiments.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np


def get_dataset(dataset_name, train=True, download=True, transform=None):
    """
    Get dataset by name.
    
    Args:
        dataset_name: Name of the dataset ('cifar10' or 'celeba')
        train: Whether to load the training or test set
        download: Whether to download the dataset
        transform: Optional transform to apply to the dataset
    
    Returns:
        torch.utils.data.Dataset: The requested dataset
    """
    # Set up default transforms if none provided
    if transform is None:
        if dataset_name.lower() == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif dataset_name.lower() == 'celeba':
            transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize(64),
                transforms.ToTensor(),
            ])
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Get the requested dataset
    if dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data',
            train=train,
            download=download,
            transform=transform
        )
    elif dataset_name.lower() == 'celeba':
        split = 'train' if train else 'valid'
        dataset = datasets.CelebA(
            root='./data',
            split=split,
            download=download,
            transform=transform
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
    
    return dataset


def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2):
    """
    Create a DataLoader for a dataset.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size for the data loader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for loading data
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def prepare_data(config, dataset_name, train=True):
    """
    Prepare dataset and dataloader based on configuration.
    
    Args:
        config: Configuration dictionary containing dataset parameters
        dataset_name: Name of the dataset
        train: Whether to load training or test set
    
    Returns:
        tuple: (dataset, dataloader)
    """
    # Get dataset configuration
    dataset_config = config['DATASETS'].get(dataset_name.lower(), {})
    batch_size = dataset_config.get('batch_size', 64)
    
    # Get dataset and create dataloader
    dataset = get_dataset(dataset_name, train=train)
    dataloader = get_dataloader(dataset, batch_size=batch_size)
    
    print(f"Prepared {dataset_name} dataset: {len(dataset)} samples, "
          f"batch size: {batch_size}, batches: {len(dataloader)}")
    
    return dataset, dataloader
