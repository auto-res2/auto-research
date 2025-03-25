import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_dataloader(config, resolution=None, train=True):
    """
    Create a dataloader for the experiment.
    
    Args:
        config: Configuration dictionary
        resolution: Optional specific resolution (overrides config)
        train: Whether to load training data (True) or test data (False)
        
    Returns:
        DataLoader object
    """
    # Set resolution from config if not provided
    if resolution is None:
        # Use first resolution from config if not specified
        resolution = config['image_resolutions'][0]
    
    # Data transform: resize to specified resolution and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    
    # Print status
    print(f"Loading {'training' if train else 'test'} data at resolution {resolution}x{resolution}")
    
    # Create dataset
    data_dir = config['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    dataset = datasets.CIFAR10(
        root=data_dir, 
        train=train, 
        download=True, 
        transform=transform
    )
    
    # Use smaller batch size for quick test if specified
    batch_size = config['quick_test_batch_size'] if config.get('quick_test', False) else config['batch_size']
    
    # Create dataloader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train,
        num_workers=config['num_workers']
    )
    
    print(f"Created dataloader with {len(dataset)} samples, batch size {batch_size}")
    return loader
