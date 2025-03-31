"""
Data preprocessing for LRE-CDT experiment.
This file contains the dataset class and data loading utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import random
import numpy as np

class VITONHD(Dataset):
    """
    Dummy implementation of VITONHD dataset for virtual try-on.
    In a real experiment, this would load actual images and masks.
    """
    def __init__(self, root_dir, transform=None, subset=None):
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset  # 'challenging', 'efficiency', etc.
        self.length = 16  # For testing, we have only 16 samples
        
        os.makedirs(root_dir, exist_ok=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.rand(3, 256, 256)
        garment_mask = (torch.rand(1, 256, 256) > 0.5).float()
        if self.transform:
            image = self.transform(image)
            garment_mask = self.transform(garment_mask)
        return image, garment_mask

def get_dataloader(config, subset=None):
    """
    Create and return a dataloader based on configuration.
    
    Args:
        config: Configuration dictionary
        subset: Optional subset name ('challenging', 'efficiency', etc.)
    
    Returns:
        DataLoader instance
    """
    transform = transforms.Compose([
        transforms.Resize((config["dataset"]["image_size"], config["dataset"]["image_size"]))
    ])
    
    dataset = VITONHD(config["dataset"]["root_dir"], transform=transform, subset=subset)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    
    return dataloader
