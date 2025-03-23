"""
Data preprocessing module for A2Diff experiments.
"""
import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

class DegradedCIFAR10(datasets.CIFAR10):
    """
    Custom CIFAR10 dataset that can apply degradation transforms to images.
    """
    def __init__(self, root, degrade_prob=0.5, transform_clean=None, **kwargs):
        super(DegradedCIFAR10, self).__init__(root, **kwargs)
        self.degrade_prob = degrade_prob
        self.degradation_transform = transforms.Compose([
            transforms.ToPILImage() if kwargs.get("transform") is not None else lambda x: x,
            transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_clean = transform_clean
        
    def __getitem__(self, index):
        image, target = super(DegradedCIFAR10, self).__getitem__(index)
        
        # Convert to PIL if it's a tensor
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            
        # Decide whether to degrade
        if np.random.random() < self.degrade_prob:
            image = self.degradation_transform(image)
            degradation_flag = True
        else:
            if self.transform_clean:
                image = self.transform_clean(image)
            degradation_flag = False
        
        return image, target, degradation_flag

def get_transforms():
    """
    Get the transforms for the dataset.
    """
    transform_basic = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return transform_basic

def load_dataset(config, subset_size=None):
    """
    Load the dataset for the experiment.
    
    Args:
        config: Experiment configuration
        subset_size: Number of samples to use (for testing)
        
    Returns:
        dataloader: DataLoader for the dataset
    """
    transform = get_transforms()
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Load dataset
    if config['data']['dataset'] == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', 
                                   download=True, 
                                   transform=transform)
        
        # Use a subset for testing if specified
        if subset_size is not None:
            dataset = Subset(dataset, indices=range(subset_size))
            
        dataloader = DataLoader(dataset, 
                               batch_size=config['training']['batch_size'], 
                               shuffle=True)
    else:
        raise ValueError(f"Dataset {config['data']['dataset']} not supported")
    
    return dataloader

def load_degraded_dataset(config, subset_size=None):
    """
    Load the degraded dataset for the robustness experiment.
    
    Args:
        config: Experiment configuration
        subset_size: Number of samples to use (for testing)
        
    Returns:
        dataloader: DataLoader for the degraded dataset
    """
    transform_clean = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Load degraded dataset
    if config['data']['dataset'] == 'CIFAR10':
        degraded_dataset = DegradedCIFAR10(
            root='./data', 
            download=True, 
            train=False, 
            degrade_prob=config['data']['degrade_probability'],
            transform_clean=transform_clean
        )
        
        # Use a subset for testing if specified
        if subset_size is not None:
            degraded_dataset = Subset(degraded_dataset, indices=range(subset_size))
            
        dataloader = DataLoader(
            degraded_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False
        )
    else:
        raise ValueError(f"Dataset {config['data']['dataset']} not supported")
    
    return dataloader
