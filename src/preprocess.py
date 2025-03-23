import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import numpy as np

def get_dataset(config):
    """
    Load and preprocess dataset based on configuration
    
    Args:
        config: Configuration dictionary containing dataset parameters
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(config['model']['image_size']),
        transforms.CenterCrop(config['model']['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*config['model']['channels'], 
                             std=[0.5]*config['model']['channels'])
    ])
    
    # Load dataset
    if config['dataset']['name'].lower() == 'celeba_hq':
        # Check if dataset path exists
        if os.path.exists(config['dataset']['path']):
            dataset = datasets.ImageFolder(config['dataset']['path'], transform=transform)
        else:
            # For testing purposes, use fake data
            print(f"Warning: Dataset path {config['dataset']['path']} not found. Using fake data.")
            dataset = datasets.FakeData(size=64, 
                                      image_size=(config['model']['channels'], 
                                                 config['model']['image_size'], 
                                                 config['model']['image_size']), 
                                      transform=transform)
    elif config['dataset']['name'].lower() == 'cifar10':
        # For CIFAR-10 dataset
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {config['dataset']['name']} not supported")
    
    # Split dataset into training and validation
    val_size = int(len(dataset) * config['dataset']['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, val_loader
