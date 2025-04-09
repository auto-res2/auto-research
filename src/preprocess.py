import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class DummyDataset(Dataset):
    """Dummy dataset for testing ADFP-Diff implementation."""
    def __init__(self, size=1000, image_size=32, transform=None):
        self.size = size
        self.image_size = image_size
        self.transform = transform
        self.data = np.random.rand(size, 3, image_size, image_size).astype(np.float32)
        self.targets = np.random.randint(0, 10, size=size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.data[idx])
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def get_transforms(image_size=32):
    """Get image transformations."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return train_transform, test_transform

def get_dataloaders(batch_size=64, image_size=32, num_workers=4):
    """Create dataloaders for training and validation."""
    train_transform, test_transform = get_transforms(image_size)
    
    train_dataset = DummyDataset(size=1000, image_size=image_size, transform=train_transform)
    val_dataset = DummyDataset(size=200, image_size=image_size, transform=test_transform)
    
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
