"""Data preprocessing for RG-MDS experiments."""

import torch
from torchvision import transforms, datasets
import os

def load_dataset(name='VOC', max_samples=None, data_dir='./data'):
    """
    Load the dataset for segmentation experiments.
    
    Args:
        name (str): Dataset name, currently supports 'VOC'
        max_samples (int, optional): Maximum number of samples to use
        data_dir (str): Directory to store dataset
        
    Returns:
        torch.utils.data.Dataset: Dataset for segmentation
    """
    print(f"Loading dataset {name}...")
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = datasets.VOCSegmentation(root=data_dir, year='2012', image_set='val', download=True,
                                      transform=preprocess, target_transform=preprocess)
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
        print(f"Using subset of {max_samples} samples.")
    print(f"Loaded {len(dataset)} samples.")
    return dataset
