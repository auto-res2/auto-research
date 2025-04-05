"""
Utility functions for dataset handling and data augmentation.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os

def create_dataloaders(dataset_train, dataset_val, batch_size=16, num_workers=4):
    """
    Create dataloaders for training and validation datasets.
    
    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        
    Returns:
        dataloaders: Dictionary containing training and validation dataloaders
    """
    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    return dataloaders

def augment_point_cloud(points, rotation=True, scaling=True, noise=True):
    """
    Apply data augmentation to point cloud.
    
    Args:
        points: Point cloud data of shape (N, 3)
        rotation: Whether to apply random rotation
        scaling: Whether to apply random scaling
        noise: Whether to apply random noise
        
    Returns:
        aug_points: Augmented point cloud
    """
    aug_points = np.copy(points)
    
    if rotation:
        theta = random.uniform(0, 2*np.pi)
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,               0,             1]
        ])
        aug_points = aug_points.dot(rot_mat.T)
        
    if scaling:
        scale_factor = random.uniform(0.8, 1.2)
        aug_points *= scale_factor
        
    if noise:
        noise_val = np.random.normal(0, 0.01, size=aug_points.shape)
        aug_points += noise_val
        
    return aug_points

class ModelNet40Dataset(Dataset):
    """
    Dataset for ModelNet40 point cloud classification.
    
    Each sample returns a (N,3) tensor (points) and an integer label.
    For this implementation, we generate random data. In a real-world
    scenario, you would load actual ModelNet40 data.
    """
    def __init__(self, split='train', augment=False, num_samples=1000, num_points=1024):
        self.split = split
        self.augment = augment
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = 40  # ModelNet40 has 40 categories
        
        np.random.seed(42 if split=='train' else 24)
        
        self.data = [np.random.rand(self.num_points, 3).astype(np.float32) for _ in range(num_samples)]
        self.labels = [np.random.randint(0, self.num_classes) for _ in range(num_samples)]
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        points = self.data[idx]
        label = self.labels[idx]
        
        if self.augment:
            points = augment_point_cloud(points)
            
        return torch.tensor(points, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        
class ShapeNetDataset(Dataset):
    """
    Dataset for ShapeNet part segmentation.
    
    Each sample returns a (N,3) tensor (points) and a segmentation label of shape (N,).
    For this implementation, we generate random data. In a real-world
    scenario, you would load actual ShapeNet data.
    """
    def __init__(self, split='train', augment=False, num_samples=500, num_points=2048):
        self.split = split
        self.augment = augment
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = 16  # dummy number of segmentation classes
        
        np.random.seed(100 if split=='train' else 50)
        
        self.data = [np.random.rand(self.num_points, 3).astype(np.float32) for _ in range(num_samples)]
        self.labels = [np.random.randint(0, self.num_classes, size=(self.num_points,)) for _ in range(num_samples)]
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        points = self.data[idx]
        labels = self.labels[idx]
        
        if self.augment:
            points = augment_point_cloud(points)
            
        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
