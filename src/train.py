"""
Model definition and training functionality for the SCND experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class MaskedAttentionBlock(nn.Module):
    """
    A masked-attention block that injects a semantic mask into the feature map.
    """
    def __init__(self, in_channels):
        super(MaskedAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, feature_map, semantic_mask):
        if semantic_mask.shape[1] != feature_map.shape[1]:
            semantic_mask = semantic_mask[:, 0:1, :, :].expand(-1, feature_map.shape[1], -1, -1)
        guided_feature = feature_map * semantic_mask
        return self.conv(guided_feature)

class DiffusionModel(nn.Module):
    """
    A diffusion network that optionally uses a Masked-Attention Guidance module.
    """
    def __init__(self, use_masked_attn=True):
        super(DiffusionModel, self).__init__()
        self.use_masked_attn = use_masked_attn
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        if self.use_masked_attn:
            self.masked_attn = MaskedAttentionBlock(64)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x, semantic_mask=None):
        x = self.encoder(x)
        if self.use_masked_attn and (semantic_mask is not None):
            x = self.masked_attn(x, semantic_mask)
        x = self.decoder(x)
        return x

def diffusion_process(model, x, semantic_mask, num_steps=10):
    """
    Simulated diffusion process that applies progressive refinement via model forward passes.
    
    Args:
        model: The diffusion model to use
        x: Initial input tensor
        semantic_mask: Semantic mask for guidance
        num_steps: Number of diffusion steps
        
    Returns:
        Tuple of (intermediate_outputs, loss_curve)
    """
    intermediate_outputs = []
    loss_curve = []
    current = x
    for step in range(num_steps):
        current = model(current, semantic_mask)
        intermediate_outputs.append(current)
        loss = F.mse_loss(current * semantic_mask, semantic_mask)
        loss_curve.append(loss.item())
    return intermediate_outputs, loss_curve

def sds_loss_fn(generated_output, target, attention_map, semantic_mask, alpha=0.1):
    """
    Compute the SDS loss with spatial mask alignment.
    
    Args:
        generated_output: Output from the model
        target: Ground-truth data
        attention_map: Attention map
        semantic_mask: Ground-truth semantic mask
        alpha: Weighting factor for spatial alignment
        
    Returns:
        Combined loss value
    """
    base_loss = F.mse_loss(generated_output, target)
    spatial_loss = F.mse_loss(attention_map, semantic_mask)
    total_loss = base_loss + alpha * spatial_loss
    return total_loss
