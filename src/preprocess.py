"""
Preprocessing module for the Latent-Integrated Fingerprint Diffusion (LIFD) method.

This module provides functionality for:
1. Loading and preprocessing image data
2. Simulating adversarial attacks on images
3. Preparing data for model training and evaluation
"""

import io
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter
from pathlib import Path

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def add_gaussian_noise(image_tensor, mean=0.0, std=0.1):
    """
    Add Gaussian noise to a tensor image.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise
        
    Returns:
        torch.Tensor: Noisy image tensor
    """
    noise = torch.randn_like(image_tensor) * std + mean
    return torch.clamp(image_tensor + noise, 0, 1)

def apply_blur(image, radius=2):
    """
    Apply Gaussian blur using PIL.
    
    Args:
        image (PIL.Image): Input PIL image
        radius (int): Blur radius
        
    Returns:
        PIL.Image: Blurred image
    """
    return image.filter(ImageFilter.GaussianBlur(radius))

def jpeg_compression(image, quality=30):
    """
    Simulate JPEG compression by saving and reloading an image.
    
    Args:
        image (PIL.Image): Input PIL image
        quality (int): JPEG quality (0-100)
        
    Returns:
        PIL.Image: Compressed image
    """
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def simulate_attacks(image_tensor):
    """
    Simulate adversarial attacks including blur, JPEG, and noise.
    Returns two attacked versions of the input image tensor.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor
        
    Returns:
        tuple: Two attacked versions of the input image tensor
            - attacked_image: Blurred and JPEG compressed image
            - attacked_noisy: Image with Gaussian noise
    """
    unloader = transforms.ToPILImage()
    image_pil = unloader(torch.clamp(image_tensor.cpu(), 0, 1))
    
    image_blur = apply_blur(image_pil, radius=2)
    image_jpeg = jpeg_compression(image_blur, quality=30)
    
    loader = transforms.ToTensor()
    attacked_image = loader(image_jpeg)
    
    attacked_noisy = add_gaussian_noise(image_tensor.clone(), std=0.05)
    
    return attacked_image, attacked_noisy


def create_transform(image_size=64, augment=True):
    """
    Create a transform pipeline for preprocessing images.
    
    Args:
        image_size (int): Target image size
        augment (bool): Whether to apply data augmentation
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform

def generate_dummy_data(batch_size=16, image_size=64, num_batches=10):
    """
    Generate dummy data for testing and development.
    
    Args:
        batch_size (int): Number of images per batch
        image_size (int): Size of the images
        num_batches (int): Number of batches to generate
        
    Returns:
        list: List of batches, where each batch is a tensor of shape [batch_size, 3, image_size, image_size]
    """
    print(f"Generating {num_batches} batches of dummy data with batch size {batch_size}")
    data = []
    for i in range(num_batches):
        batch = torch.rand(batch_size, 3, image_size, image_size)
        data.append(batch)
    return data

def generate_fingerprints(num_users=10, fingerprint_dim=128):
    """
    Generate random fingerprints for users.
    
    Args:
        num_users (int): Number of users
        fingerprint_dim (int): Dimension of the fingerprint
        
    Returns:
        torch.Tensor: Tensor of shape [num_users, fingerprint_dim] containing binary fingerprints
    """
    print(f"Generating {num_users} fingerprints with dimension {fingerprint_dim}")
    fingerprints = torch.randint(0, 2, (num_users, fingerprint_dim), dtype=torch.float32)
    return fingerprints

def prepare_data_for_training(data_dir=None, batch_size=16, image_size=64, use_dummy=True, num_users=10, fingerprint_dim=128):
    """
    Prepare data for training the LIFD model.
    
    Args:
        data_dir (str): Directory containing the image data
        batch_size (int): Batch size
        image_size (int): Target image size
        use_dummy (bool): Whether to use dummy data
        num_users (int): Number of users for fingerprint generation
        fingerprint_dim (int): Dimension of the fingerprints
        
    Returns:
        tuple: 
            - train_data: Training data
            - val_data: Validation data
            - fingerprints: User fingerprints
    """
    if use_dummy or data_dir is None:
        print("Using dummy data for training")
        all_data = generate_dummy_data(batch_size, image_size, num_batches=20)
        
        train_data = all_data[:16]
        val_data = all_data[16:]
        
        fingerprints = generate_fingerprints(num_users=num_users, fingerprint_dim=fingerprint_dim)
        
        return train_data, val_data, fingerprints
    else:
        raise NotImplementedError("Real data loading not implemented yet")

def preprocess_data(config=None):
    """
    Main preprocessing function that prepares data for the LIFD model.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        dict: Preprocessed data and metadata
    """
    if config is None:
        config = {
            'seed': 42,
            'batch_size': 16,
            'image_size': 64,
            'use_dummy': True,
            'data_dir': None,
            'fingerprint_dim': 128,
            'num_users': 10
        }
    
    set_seed(config['seed'])
    
    train_data, val_data, fingerprints = prepare_data_for_training(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        use_dummy=config['use_dummy'],
        num_users=config['num_users'],
        fingerprint_dim=config['fingerprint_dim']
    )
    
    preprocessed_data = {
        'train_data': train_data,
        'val_data': val_data,
        'fingerprints': fingerprints,
        'config': config
    }
    
    print("Data preprocessing completed successfully")
    return preprocessed_data

if __name__ == "__main__":
    config = {
        'seed': 42,
        'batch_size': 16,
        'image_size': 64,
        'use_dummy': True,
        'data_dir': None,
        'fingerprint_dim': 128,
        'num_users': 10
    }
    
    preprocessed_data = preprocess_data(config)
    print(f"Train data: {len(preprocessed_data['train_data'])} batches")
    print(f"Validation data: {len(preprocessed_data['val_data'])} batches")
    print(f"Fingerprints shape: {preprocessed_data['fingerprints'].shape}")
