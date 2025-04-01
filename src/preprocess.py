"""
Data preprocessing functionality for the SCND experiment.
"""

import torch
import numpy as np

def get_dummy_data(batch_size=1, channels=3, height=256, width=256):
    """
    Create a dummy RGB image tensor and a corresponding semantic mask.
    The image tensor is random and the semantic mask is a binary mask with smooth edges.
    
    Args:
        batch_size: Number of samples in the batch
        channels: Number of image channels
        height: Image height
        width: Image width
        
    Returns:
        Tuple of (image, semantic_mask) as torch tensors
    """
    image = torch.rand(batch_size, channels, height, width)
    yy, xx = torch.meshgrid(torch.linspace(-1,1,height), torch.linspace(-1,1,width), indexing="ij")
    radius = torch.sqrt(xx**2 + yy**2)
    mask = (radius < 0.8).float()  # binary mask values
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)
    return image, mask
