"""
Data preprocessing for PurifyCov++ experiments.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from utils.data import get_test_loader

def preprocess_data(batch_size=32, num_workers=0, quick_test=False):
    """
    Preprocess data for experiments.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loaders
        quick_test: If True, use a small subset of the data for testing
        
    Returns:
        test_loader: PyTorch DataLoader for the test dataset
    """
    print("Preprocessing data...")
    
    test_loader = get_test_loader(batch_size, num_workers, quick_test)
    
    print(f"Data preprocessing complete. Test set: {len(test_loader.dataset)} samples")
    
    return test_loader

if __name__ == "__main__":
    test_loader = preprocess_data(quick_test=True)
    print("Data preview:")
    images, labels = next(iter(test_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
