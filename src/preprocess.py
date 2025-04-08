"""
Preprocessing script for ADNLCC method.
This script handles the loading and preprocessing of data for the diffusion model.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

def get_dataloaders(batch_size=64, dataset_name="CIFAR10", add_noise=True, noise_level=0.5):
    """
    Create and return dataloaders for training and testing.
    
    Args:
        batch_size (int): Batch size for the dataloaders
        dataset_name (str): Name of the dataset to use, currently supports CIFAR10
        add_noise (bool): Whether to add noise to the dataset for ADNLCC training
        noise_level (float): Level of noise to add, between 0.0 and 1.0
        
    Returns:
        tuple: (train_loader, test_loader) - DataLoader objects for training and testing
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    if add_noise:
        train_dataset = NoisyDataset(train_dataset, noise_level=noise_level)
        test_dataset = NoisyDataset(test_dataset, noise_level=noise_level)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return train_loader, test_loader

class NoisyDataset(Dataset):
    """Dataset wrapper that adds noise to the images."""
    def __init__(self, dataset, noise_level=0.5):
        self.dataset = dataset
        self.noise_level = noise_level
        
    def __getitem__(self, index):
        img, label = self.dataset[index]
        noise = torch.randn_like(img) * self.noise_level
        img = img + noise
        img = torch.clamp(img, -1.0, 1.0)
        return img, label
    
    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
