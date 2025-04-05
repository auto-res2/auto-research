"""
Preprocessing module for IDRR-GAR experiments.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os

class DynamicScenesDataset(Dataset):
    """
    Dataset for dynamic scenes with depth maps.
    For demonstration, this creates dummy data. 
    In a real implementation, this would load actual image and depth data.
    """
    def __init__(self, num_samples=10, transform=None, img_height=256, img_width=320):
        """
        Initialize the dataset.
        
        Args:
            num_samples: Number of samples to generate
            transform: Transforms to apply to the images
            img_height: Height of the images
            img_width: Width of the images
        """
        self.num_samples = num_samples
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate a random sample (image and depth map).
        
        Args:
            idx: Index of the sample
            
        Returns:
            image: Random image tensor (3, H, W)
            gt_depth: Random depth map tensor (1, H, W)
        """
        image = torch.rand(3, self.img_height, self.img_width)
        
        gt_depth = torch.rand(1, self.img_height, self.img_width)
        
        if self.transform:
            image = self.transform(image)
            
        return image, gt_depth

def get_data_loaders(config):
    """
    Create data loaders for training and evaluation.
    
    Args:
        config: Configuration parameters
        
    Returns:
        train_loader: DataLoader for training
        eval_loader: DataLoader for evaluation
    """
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
    ])
    
    train_dataset = DynamicScenesDataset(
        num_samples=20, 
        transform=transform,
        img_height=config.IMAGE_HEIGHT,
        img_width=config.IMAGE_WIDTH
    )
    
    eval_dataset = DynamicScenesDataset(
        num_samples=10, 
        transform=transform,
        img_height=config.IMAGE_HEIGHT,
        img_width=config.IMAGE_WIDTH
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, eval_loader
