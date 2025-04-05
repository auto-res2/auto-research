"""
ACAG-OVS Data Preprocessing Module

This module handles data loading and preprocessing for the ACAG-OVS experiments.
"""

import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm

class DummyDataset(torch.utils.data.Dataset):
    """
    Creates a synthetic dataset for ACAG-OVS experiments.
    
    This dataset is used for simulating cross-attention maps and segmentation masks
    when real data is not available or for quick testing of the models.
    """
    def __init__(self, num_samples=10, image_size=(3, 64, 64), data_dir=None):
        """
        Initialize the dummy dataset.
        
        Args:
            num_samples (int): Number of synthetic samples to generate
            image_size (tuple): Size of the synthetic images (C, H, W)
            data_dir (str): Directory to save/load data (not used for synthetic data)
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.data_dir = data_dir

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(self.image_size)  # random image
        mask = torch.randint(0, 2, self.image_size[1:])
        return image, mask

def get_data_loader(batch_size=2, num_samples=50, data_dir=None):
    """
    Create a data loader for the experiments.
    
    Args:
        batch_size (int): Batch size for the data loader
        num_samples (int): Number of samples in the dataset
        data_dir (str): Directory where data is stored (if using real data)
        
    Returns:
        torch.utils.data.DataLoader: Data loader for the experiments
    """
    dataset = DummyDataset(num_samples=num_samples, data_dir=data_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
