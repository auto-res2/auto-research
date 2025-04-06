"""
ClusterCloak: Preprocessing Module

This module contains functions for preprocessing image data for the ClusterCloak experiments.
It handles loading dummy data, applying transforms, and creating datasets for experiments.
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    """Dataset class for generating random image data for experiments."""
    def __init__(self, num_samples, transform=None, img_size=224):
        self.transform = transform
        self.img_size = img_size
        self.data = [np.uint8(np.random.rand(img_size, img_size, 3) * 255) 
                    for _ in range(num_samples)]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img
