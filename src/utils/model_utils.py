"""
Model utility functions for LuminoDiff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from typing import Tuple, Optional


def disentanglement_loss(content_latent: torch.Tensor, 
                        brightness_latent: torch.Tensor) -> torch.Tensor:
    """
    Penalize cross-correlation between content and brightness latents.
    
    Args:
        content_latent (torch.Tensor): Content latent representation
        brightness_latent (torch.Tensor): Brightness latent representation
        
    Returns:
        torch.Tensor: Disentanglement loss
    """
    b, c, h, w = content_latent.size()
    content_flat = content_latent.view(b, c, -1)
    bright_flat = brightness_latent.view(b, brightness_latent.size(1), -1)
    corr = torch.mean(torch.abs(torch.bmm(content_flat, 
                                          bright_flat.transpose(1, 2))))
    return corr


def brightness_histogram_kl(true_img: np.ndarray, 
                           gen_img: np.ndarray, 
                           num_bins: int = 50) -> float:
    """
    Calculate KL divergence between brightness histograms.
    
    Args:
        true_img (np.ndarray): Ground truth image
        gen_img (np.ndarray): Generated image
        num_bins (int): Number of bins for histogram
        
    Returns:
        float: KL divergence score
    """
    hist_true, _ = np.histogram(true_img.flatten(), bins=num_bins, 
                                range=(0, 1), density=True)
    hist_gen, _ = np.histogram(gen_img.flatten(), bins=num_bins, 
                              range=(0, 1), density=True)
    epsilon = 1e-10
    kl_div = float(entropy(hist_true + epsilon, hist_gen + epsilon))
    return kl_div


def brightness_loss(noise_pred: torch.Tensor, 
                   target_noise: torch.Tensor, 
                   variant: str = 'A', 
                   weight: float = 1.0) -> torch.Tensor:
    """
    Custom brightness loss based on variant selection.
    
    Args:
        noise_pred (torch.Tensor): Predicted noise
        target_noise (torch.Tensor): Target noise
        variant (str): Loss variant type (A, B, or C)
        weight (float): Weight for the loss
        
    Returns:
        torch.Tensor: Computed loss
    """
    mse_loss = F.mse_loss(noise_pred, target_noise)
    if variant == 'A':
        refinement_loss = 0.5 * F.mse_loss(noise_pred, target_noise)
        return weight * (mse_loss + refinement_loss)
    elif variant == 'B':
        return weight * mse_loss
    elif variant == 'C':
        reg_loss = weight * torch.mean(torch.abs(noise_pred - target_noise))
        return reg_loss
    else:
        return mse_loss
