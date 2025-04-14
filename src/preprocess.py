"""
Data preprocessing for the Progressive Brightness Distillation Diffusion experiment.
"""

import os
import random
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config.pbd_diffusion_config import (
    SEED,
    DATASET_NAME,
    BATCH_SIZE,
    DOWNLOAD,
)

def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return

def get_cifar10_datasets():
    """
    Create CIFAR10 datasets with original and biased brightness transformations.
    Returns datasets for training and testing.
    """
    transform_original = transforms.Compose([transforms.ToTensor()])
    
    transform_biased = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * float(np.random.uniform(0.5, 1.5)))
    ])
    
    train_dataset = CIFAR10(root='./data', train=True, download=DOWNLOAD, transform=transform_biased)
    test_dataset = CIFAR10(root='./data', train=False, download=DOWNLOAD, transform=transform_biased)
    
    train_gt_dataset = CIFAR10(root='./data', train=True, download=DOWNLOAD, transform=transform_original)
    test_gt_dataset = CIFAR10(root='./data', train=False, download=DOWNLOAD, transform=transform_original)
    
    return train_dataset, test_dataset, train_gt_dataset, test_gt_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size=BATCH_SIZE):
    """Create DataLoader objects for training and testing."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def prepare_data():
    """
    Prepare all datasets and data loaders for the experiment.
    Returns all necessary data loaders.
    """
    set_seed()
    
    train_dataset, test_dataset, train_gt_dataset, test_gt_dataset = get_cifar10_datasets()
    
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset)
    train_gt_loader, test_gt_loader = get_data_loaders(train_gt_dataset, test_gt_dataset)
    
    return train_loader, test_loader, train_gt_loader, test_gt_loader
