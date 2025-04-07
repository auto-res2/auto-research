
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def load_cifar10(batch_size=32, download=True):
    """
    Load CIFAR-10 dataset and prepare data loaders for training and testing.
    
    Args:
        batch_size (int): Batch size for data loaders
        download (bool): Whether to download the dataset if not already downloaded
        
    Returns:
        tuple: (train_loader, test_loader) - PyTorch data loaders for training and testing
    """
    os.makedirs('./data', exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=download, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=download, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
