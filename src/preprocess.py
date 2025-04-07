"""
Preprocessing module for GCAD experiments.

This module contains functions for loading and preprocessing data
for the Geometrically Consistent Ambient Diffusion experiments.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np

def get_cifar10_data(data_dir='./data', batch_size=128, subset_size=None):
    """
    Load CIFAR-10 dataset for training and evaluation.
    
    Args:
        data_dir: Directory to store the dataset
        batch_size: Batch size for dataloaders
        subset_size: If provided, creates a subset of the dataset with every subset_size-th sample
        
    Returns:
        train_dataset, train_loader, val_dataset, val_loader
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    if subset_size is not None:
        subset_indices = list(range(0, len(train_dataset), subset_size))
        train_dataset = Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, train_loader, val_dataset, val_loader

def corrupt_images(images, noise_type="gaussian", noise_level=0.1):
    """
    Add corruptions to images: either additive Gaussian noise or structured linear (stripe) corruption.
    
    Args:
        images: Tensor of images to corrupt
        noise_type: Type of corruption, either "gaussian" or "linear"
        noise_level: Level of Gaussian noise (ignored for linear corruption)
        
    Returns:
        Corrupted images
    """
    if noise_type == "gaussian":
        noise = noise_level * torch.randn_like(images)
        return images + noise
    elif noise_type == "linear":
        corrupted = images.clone()
        _, _, H, _ = corrupted.shape
        start, end = H // 3, H // 3 + max(1, H // 10)
        corrupted[:, :, start:end, :] = 0.0
        return corrupted
    else:
        return images
