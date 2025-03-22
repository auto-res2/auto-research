"""
Data preprocessing for SBDT experiments.
"""

import torch
import numpy as np
from torchvision import datasets, transforms
import os

def load_dataset(dataset_name, root='./data', train=True, download=True, transform=None, subset_size=None):
    """
    Load a dataset for SBDT experiments.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CIFAR10')
        root: Root directory for dataset
        train: Whether to load training set
        download: Whether to download the dataset
        transform: Transforms to apply
        subset_size: Optional size of subset to use
        
    Returns:
        Dataset object
    """
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    
    # Create data directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    # Load dataset based on name
    if dataset_name.upper() == 'CIFAR10':
        dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create subset if specified
    if subset_size is not None and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    return dataset

def create_dataloaders(dataset, batch_size=32, shuffle=True):
    """
    Create DataLoader from dataset.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_flattened_features(dataloader, device="cpu"):
    """
    Extract flattened features from dataloader for anomaly detection.
    
    Args:
        dataloader: DataLoader containing images
        device: Device to use for computation
        
    Returns:
        Numpy array of flattened features
    """
    features = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            flat = data.view(data.size(0), -1).cpu().numpy()
            features.append(flat)
    return np.concatenate(features, axis=0)
