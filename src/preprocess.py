"""
Preprocessing module for the NTEC-G experiment.
Handles data loading and preparation for the diffusion model experiments.
"""

import torch
import torchvision
from torchvision import transforms
import numpy as np

def load_dummy_data(batch_size=64, latent_dim=128):
    """
    Creates dummy data for testing the NTEC-G method.
    
    Args:
        batch_size: Number of samples in a batch
        latent_dim: Dimension of the latent space
        
    Returns:
        torch.Tensor: Random data tensor of shape (batch_size, latent_dim)
    """
    return torch.randn(batch_size, latent_dim)

def prepare_visualization_data(latents, reshape_to_image=False):
    """
    Prepares latent data for visualization.
    
    Args:
        latents: Latent representations
        reshape_to_image: Whether to reshape latents to image-like format
        
    Returns:
        torch.Tensor: Processed data ready for visualization
    """
    if reshape_to_image:
        batch_size = latents.shape[0]
        samples = []
        
        for i in range(batch_size):
            state_flat = latents[i].view(-1)
            repeated = state_flat.repeat((3072 // state_flat.shape[0] + 1))[:3072]
            image = repeated.view(3, 32, 32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            samples.append(image)
            
        return torch.stack(samples)
    
    return latents
