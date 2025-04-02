"""
LuminoDiff Training Module

This module implements training functions for the LuminoDiff model.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional

from src.utils.model_utils import disentanglement_loss, brightness_loss


def training_step_brightness(model: nn.Module, brightness_branch: nn.Module, 
                           images: torch.Tensor, noise_target: torch.Tensor, 
                           variant: str = 'A') -> float:
    """
    A simple training step that uses the dual latent encoder and brightness branch.
    
    Args:
        model (nn.Module): Dual latent encoder model
        brightness_branch (nn.Module): Brightness branch model
        images (torch.Tensor): Input images
        noise_target (torch.Tensor): Target noise tensor
        variant (str): Brightness loss variant (A, B, or C)
        
    Returns:
        float: Loss value
    """
    model.train()
    brightness_branch.train()
    
    content_latent, brightness_latent = model(images)
    noise_pred = brightness_branch(brightness_latent)
    
    b_loss = brightness_loss(noise_pred, noise_target, variant=variant, weight=1.0)
    
    reconstruction_loss = torch.tensor(0.0, device=images.device)
    
    total_loss = reconstruction_loss + b_loss
    
    
    return total_loss.item()


def run_ablation_study(base_encoder: nn.Module, dual_encoder: nn.Module, 
                      dummy_images: torch.Tensor, device: str = 'cuda') -> Dict:
    """
    Run ablation study comparing base and dual-latent encoders.
    
    Args:
        base_encoder (nn.Module): Base encoder model
        dual_encoder (nn.Module): Dual-latent encoder model
        dummy_images (torch.Tensor): Input images
        device (str): Device to run on ('cuda' or 'cpu')
        
    Returns:
        Dict: Results of the ablation study
    """
    base_encoder.train()
    dual_encoder.train()
    
    dummy_images = dummy_images.to(device)
    base_encoder = base_encoder.to(device)
    dual_encoder = dual_encoder.to(device)
    
    z_base = base_encoder(dummy_images)
    
    content_latent, brightness_latent = dual_encoder(dummy_images)
    
    d_loss = disentanglement_loss(content_latent, brightness_latent)
    
    results = {
        'base_latent_shape': z_base.shape,
        'content_latent_shape': content_latent.shape,
        'brightness_latent_shape': brightness_latent.shape,
        'disentanglement_loss': d_loss.item()
    }
    
    return results
