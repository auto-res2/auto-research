"""
Preprocessing module for SPCDD MRI Super-Resolution.

This module handles data loading, preprocessing, and anatomical prior extraction.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SyntheticMRIDataset(Dataset):
    """Synthetic MRI dataset for testing the SPCDD method."""
    
    def __init__(self, num_samples=20, image_size=(1, 64, 64)):
        """
        Initialize the synthetic MRI dataset.
        
        Args:
            num_samples: Number of synthetic samples to generate
            image_size: Size of the images (channels, height, width)
        """
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a synthetic sample pair of 1.5T and 7T MRI images."""
        img_15T = torch.rand(self.image_size)
        
        noise = 0.1 * torch.rand(self.image_size)
        structured_pattern = 0.2 * torch.sin(
            torch.linspace(0, 3*np.pi, self.image_size[1]).unsqueeze(0).unsqueeze(0).repeat(
                self.image_size[0], 1, self.image_size[2]
            )
        )
        target_7T = torch.clamp(img_15T + noise + structured_pattern, 0, 1)
        
        meta = 0
        
        return img_15T, target_7T, meta

class AnatomyExtractor(nn.Module):
    """
    Lightweight anatomy extractor network to generate anatomical templates 
    and segmentation masks from 1.5T MRI images.
    """
    
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1):
        """
        Initialize the anatomy extractor network.
        
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
            out_channels: Number of output channels
        """
        super(AnatomyExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1), 
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1), 
            nn.Sigmoid()  # Output a segmentation-like map
        )
    
    def forward(self, x):
        """Forward pass through the anatomy extractor."""
        features = self.encoder(x)
        anatomy_map = self.decoder(features)
        return anatomy_map

def preprocess_data(config):
    """
    Preprocess data for the SPCDD method.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    set_seed(config.random_seed)
    
    train_dataset = SyntheticMRIDataset(
        num_samples=config.synthetic_dataset_size,
        image_size=(1, config.image_size, config.image_size)
    )
    
    val_dataset = SyntheticMRIDataset(
        num_samples=config.synthetic_dataset_size // 5,  # Smaller validation set
        image_size=(1, config.image_size, config.image_size)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader
