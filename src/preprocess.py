"""Data preprocessing for CSTD experiments."""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import sys

# Add the repository root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import cstd_config as cfg

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_trigger_patch(image_shape, patch_size=5):
    """Create a simple white square patch trigger in the top-left corner.
    
    Args:
        image_shape: The shape of the image (batch_size, channels, height, width)
        patch_size: Size of the square patch
        
    Returns:
        trigger: The trigger pattern
        mask: The mask indicating the trigger location
    """
    trigger = torch.zeros(image_shape)
    mask = torch.zeros(image_shape)
    trigger[:, :, :patch_size, :patch_size] = 1.0
    mask[:, :, :patch_size, :patch_size] = 1.0
    return trigger, mask

def implant_trigger(images, trigger, mask, ratio=0.3):
    """Implant trigger into a random fraction (ratio) of images in the batch.
    
    Args:
        images: Batch of images
        trigger: The trigger pattern
        mask: The mask indicating the trigger location
        ratio: Fraction of images to implant the trigger
        
    Returns:
        Modified images with triggers implanted
    """
    batch_size = images.size(0)
    num_implant = int(batch_size * ratio)
    if num_implant < 1:
        return images
    idx = np.random.choice(batch_size, num_implant, replace=False)
    # For each selected image, apply: image = image*(1-mask) + trigger*mask
    images_copy = images.clone()
    images_copy[idx] = images_copy[idx] * (1 - mask) + trigger * mask
    return images_copy

def add_gaussian_noise(images, sigma):
    """Add Gaussian noise to images.
    
    Args:
        images: Batch of images
        sigma: Standard deviation of the Gaussian noise
        
    Returns:
        Noisy images
    """
    noise = sigma * torch.randn_like(images)
    return images + noise

def load_data(dataset_name, batch_size, test_mode=False, test_subset_size=256):
    """Load dataset for experiments.
    
    Args:
        dataset_name: Name of the dataset ("cifar10", "cifar100", etc.)
        batch_size: Batch size for data loading
        test_mode: If True, use a small subset for testing
        test_subset_size: Size of the test subset
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset_name.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # If in test mode, use small subsets
    if test_mode:
        train_dataset = Subset(train_dataset, indices=list(range(test_subset_size)))
        test_dataset = Subset(test_dataset, indices=list(range(test_subset_size // 2)))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS)
    
    return train_loader, test_loader
