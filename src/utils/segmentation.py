"""Segmentation functions for RG-MDS."""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

def base_segmentation(img):
    """
    Base segmentation function (token-attention branch).
    In a real implementation, this would run a diffusion-based segmentation model.
    Here we simply threshold the grayscale version of the image.
    
    Args:
        img (PIL.Image): Input image
        
    Returns:
        numpy.ndarray: Binary segmentation mask
    """
    img_tensor = transforms.ToTensor()(img)
    seg_mask = (img_tensor.mean(dim=0) > 0.5).float()
    return seg_mask.numpy()

def rg_mds_segmentation(img, reference, weighting_mode='adaptive'):
    """
    RG-MDS segmentation by fusing the base token-attention segmentation
    with a reference-based segmentation. The weighting can be adaptive or fixed.
    
    Args:
        img (PIL.Image): Input image
        reference (PIL.Image): Reference image
        weighting_mode (str): 'adaptive' or 'fixed'
        
    Returns:
        numpy.ndarray: Binary segmentation mask
    """
    token_mask = base_segmentation(img)
    reference_mask = base_segmentation(reference)
    
    if weighting_mode == 'adaptive':
        token_confidence = np.mean(token_mask)  
        weight = np.clip(token_confidence, 0.3, 0.7)
    else:  # fixed weighting
        weight = 0.5
    fused_mask = weight * token_mask + (1 - weight) * reference_mask
    fused_mask = (fused_mask > 0.5).astype(np.float32)
    return fused_mask

def limited_prompt_segmentation(img, prompt="incomplete description"):
    """
    Simulates a case of limited token expressiveness.
    Here, we generate a base segmentation and then add random noise.
    
    Args:
        img (PIL.Image): Input image
        prompt (str): Prompt to use (simulated)
        
    Returns:
        numpy.ndarray: Binary segmentation mask with noise
    """
    seg = base_segmentation(img)
    noise = np.random.binomial(1, 0.1, seg.shape)
    noisy_seg = np.logical_xor(seg, noise).astype(np.float32)
    return noisy_seg
