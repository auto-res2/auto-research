#!/usr/bin/env python3
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_transforms(image_size=32):
    """
    Define the image transformations for preprocessing.
    
    Args:
        image_size (int): Size to resize images to
        
    Returns:
        transforms: Composition of image transformations
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_cifar10(batch_size=64, image_size=32, num_workers=2, data_dir='./data'):
    """
    Load the CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for data loading
        image_size (int): Size to resize images to
        num_workers (int): Number of worker threads for data loading
        data_dir (str): Directory to store dataset
        
    Returns:
        dataloader: DataLoader for the CIFAR-10 dataset
    """
    transform = get_transforms(image_size)
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    return dataloader

def preprocess_data(config):
    """
    Main preprocessing function that loads and processes data based on configuration.
    
    Args:
        config (dict): Configuration parameters for preprocessing
        
    Returns:
        dataloader: DataLoader for the preprocessed dataset
    """
    print("Starting data preprocessing...")
    dataloader = load_cifar10(
        batch_size=config.get('batch_size', 64),
        image_size=config.get('image_size', 32),
        num_workers=config.get('num_workers', 2),
        data_dir=config.get('data_dir', './data')
    )
    print(f"Preprocessing complete. DataLoader created with {len(dataloader)} batches.")
    return dataloader

if __name__ == "__main__":
    # Simple test for the preprocessing module
    test_config = {
        'batch_size': 64,
        'image_size': 32,
        'num_workers': 2,
        'data_dir': './data'
    }
    dataloader = preprocess_data(test_config)
    print(f"Test DataLoader created with {len(dataloader)} batches.")
