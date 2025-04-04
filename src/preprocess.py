"""
Data preprocessing for SAC-Seg experiments.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, Dict, List, Optional, Union


class RandomSegmentationDataset(Dataset):
    """
    A synthetic dataset for segmentation tasks.
    
    This dataset generates random images and segmentation masks for testing
    the SAC-Seg method without requiring external data.
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        num_samples: int = 100,
        num_classes: int = 16,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_size: Tuple of (height, width) for generated images
            num_samples: Number of samples in the dataset
            num_classes: Number of segmentation classes
            transform: Optional transforms to apply to images
        """
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a random image and segmentation mask.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, mask) tensors
        """
        image = np.random.rand(self.image_size[0], self.image_size[1], 3).astype(np.float32)
        
        mask = np.random.randint(0, self.num_classes, self.image_size).astype(np.int64)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            
        mask = torch.from_numpy(mask)
        return image, mask


def get_data_loaders(
    image_size: Tuple[int, int] = (512, 512),
    num_samples: int = 100,
    num_classes: int = 16,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        image_size: Tuple of (height, width) for generated images
        num_samples: Number of samples in the dataset
        num_classes: Number of segmentation classes
        batch_size: Batch size for the data loaders
        train_ratio: Ratio of data to use for training
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = RandomSegmentationDataset(
        image_size=image_size,
        num_samples=num_samples,
        num_classes=num_classes,
        transform=transform
    )
    
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader
