"""
Data loading and preprocessing utilities for PurifyCov++ experiments.
"""

import torch
import torchvision
import torchvision.transforms as transforms

def get_test_loader(batch_size=32, num_workers=0, quick_test=False):
    """
    Get CIFAR-10 test data loader.
    
    Args:
        batch_size: Batch size for the data loader
        num_workers: Number of workers for the data loader
        quick_test: If True, use a small subset of the data for testing
        
    Returns:
        test_loader: PyTorch DataLoader for the test dataset
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True,
        transform=transform
    )
    
    if quick_test:
        test_dataset.data = test_dataset.data[:batch_size]
        test_dataset.targets = test_dataset.targets[:batch_size]
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return test_loader
