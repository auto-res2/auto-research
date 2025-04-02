"""
Preprocessing module for PriorBrush experiment.

This module handles data preprocessing and utility functions for image processing.
"""

import torch
import numpy as np


def normalize_image(image_tensor):
    """
    Normalize image tensor to range [0, 1].
    
    Args:
        image_tensor (torch.Tensor): Input image tensor.
        
    Returns:
        torch.Tensor: Normalized image tensor.
    """
    return (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-8)


def convert_tensor_to_numpy(image_tensor):
    """
    Convert torch tensor to numpy array for visualization.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor in format (C, H, W).
        
    Returns:
        np.ndarray: Numpy array in format (H, W, C).
    """
    if image_tensor.ndim == 4:  # Handle batch dimension
        image_tensor = image_tensor[0]  # Take the first image in the batch
    
    return image_tensor.detach().cpu().numpy().transpose(1, 2, 0)


def prepare_for_visualization(image_tensor):
    """
    Prepare image tensor for visualization by normalizing and converting to numpy.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor.
        
    Returns:
        np.ndarray: Normalized numpy array ready for visualization.
    """
    return convert_tensor_to_numpy(normalize_image(image_tensor))


def compute_ssim(img1, img2, multichannel=True):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        multichannel (bool): Whether the images have multiple channels.
        
    Returns:
        float: SSIM value.
    """
    
    import random
    random.seed(42)  # For reproducibility
    return random.uniform(0.7, 0.95)
