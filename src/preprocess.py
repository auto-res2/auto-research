"""
LuminoDiff Preprocessing Module

This module handles data generation and preprocessing for the LuminoDiff model.
"""

import os
import torch
import random
import numpy as np
from typing import Tuple, List


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_dummy_images(batch_size: int = 4, channels: int = 3, 
                          height: int = 64, width: int = 64) -> torch.Tensor:
    """
    Create random images with values between 0 and 1. 
    For high contrast, add two modes (dark and bright images).
    
    Args:
        batch_size (int): Number of images to generate
        channels (int): Number of channels (RGB=3)
        height (int): Image height
        width (int): Image width
        
    Returns:
        torch.Tensor: Batch of generated images
    """
    images = []
    for _ in range(batch_size):
        if random.random() < 0.5:
            img = torch.rand(channels, height, width) * 0.3
        else:
            img = 0.7 + torch.rand(channels, height, width) * 0.3
        images.append(img)
    images = torch.stack(images)
    return images


def compute_brightness_channel(images: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB images to grayscale via a weighted sum
    
    Args:
        images (torch.Tensor): RGB images batch
        
    Returns:
        torch.Tensor: Brightness channel
    """
    r, g, b = images[:, 0:1, :, :], images[:, 1:2, :, :], images[:, 2:3, :, :]
    brightness = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return brightness
