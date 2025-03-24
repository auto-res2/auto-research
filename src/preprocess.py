# src/preprocess.py
import torch
import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from src.utils.data import NoisyCIFAR10, get_dataloaders

def preprocess_data(noise_levels=[0.1, 0.3, 0.6], batch_size=64):
    """
    Preprocess data by downloading CIFAR10 and preparing noisy versions
    with different noise levels.
    
    Args:
        noise_levels: List of noise levels to use
        batch_size: Batch size for dataloaders
        
    Returns:
        dataloaders: Dictionary of dataloaders with different noise levels
    """
    print("Preprocessing data...")
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Download CIFAR10 if not already downloaded
    CIFAR10(root='./data', train=True, download=True)
    CIFAR10(root='./data', train=False, download=True)
    
    # Create dataloaders for different noise levels
    dataloaders = {}
    for noise_level in noise_levels:
        print(f"Creating dataloaders with noise level {noise_level}...")
        train_loader, test_loader = get_dataloaders(batch_size=batch_size, noise_level=noise_level)
        dataloaders[noise_level] = {
            'train': train_loader,
            'test': test_loader
        }
    
    print("Data preprocessing complete.")
    return dataloaders

if __name__ == "__main__":
    # Run a simple test of the preprocessing
    dataloaders = preprocess_data(noise_levels=[0.1], batch_size=64)
    
    # Display a sample of noisy images
    for noise_level, loaders in dataloaders.items():
        train_loader = loaders['train']
        images, _ = next(iter(train_loader))
        print(f"Loaded {len(train_loader.dataset)} training images with noise level {noise_level}")
        print(f"Image batch shape: {images.shape}")
        print(f"Image range: [{images.min().item():.2f}, {images.max().item():.2f}]")
