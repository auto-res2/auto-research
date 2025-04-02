"""
LuminoDiff Evaluation Module

This module evaluates the LuminoDiff model components.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.preprocess import compute_brightness_channel
from src.utils.model_utils import brightness_histogram_kl


def compare_fusion_mechanisms(fusion_model_attn: torch.nn.Module, 
                            fusion_model_base: torch.nn.Module,
                            dummy_images: torch.Tensor,
                            device: str = 'cuda') -> Dict:
    """
    Compare attention-based fusion with baseline fusion.
    
    Args:
        fusion_model_attn (torch.nn.Module): Model with attention fusion
        fusion_model_base (torch.nn.Module): Model with baseline fusion
        dummy_images (torch.Tensor): Input images
        device (str): Device to run on ('cuda' or 'cpu')
        
    Returns:
        Dict: Comparison results
    """
    fusion_model_attn.eval()
    fusion_model_base.eval()
    
    dummy_images = dummy_images.to(device)
    fusion_model_attn = fusion_model_attn.to(device)
    fusion_model_base = fusion_model_base.to(device)
    
    with torch.no_grad():
        fused_attn, attn_weights = fusion_model_attn(dummy_images)
        fused_base, _ = fusion_model_base(dummy_images)
    
    results = {
        'attn_fused_shape': fused_attn.shape,
        'base_fused_shape': fused_base.shape,
        'has_attention_weights': attn_weights is not None,
        'attention_shape': attn_weights.shape if attn_weights is not None else None
    }
    
    return results


def evaluate_brightness_metrics(true_images: torch.Tensor, 
                              generated_images: torch.Tensor) -> Dict:
    """
    Evaluate brightness metrics between true and generated images.
    
    Args:
        true_images (torch.Tensor): Ground truth images
        generated_images (torch.Tensor): Model generated images
        
    Returns:
        Dict: Brightness evaluation metrics
    """
    true_np = true_images.detach().cpu().numpy()
    gen_np = generated_images.detach().cpu().numpy()
    
    brightness_true = compute_brightness_channel(true_images).cpu().numpy()
    brightness_gen = compute_brightness_channel(generated_images).cpu().numpy()
    
    kl_div = brightness_histogram_kl(brightness_true, brightness_gen)
    
    true_brightness_mean = np.mean(brightness_true)
    gen_brightness_mean = np.mean(brightness_gen)
    true_brightness_std = np.std(brightness_true)
    gen_brightness_std = np.std(brightness_gen)
    
    results = {
        'kl_divergence': kl_div,
        'true_brightness_mean': true_brightness_mean,
        'gen_brightness_mean': gen_brightness_mean,
        'true_brightness_std': true_brightness_std,
        'gen_brightness_std': gen_brightness_std,
        'brightness_mean_diff': abs(true_brightness_mean - gen_brightness_mean),
        'brightness_std_diff': abs(true_brightness_std - gen_brightness_std)
    }
    
    return results
