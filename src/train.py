"""
Training module for IDRR-GAR experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class BaseDepthModel(nn.Module):
    """
    Base depth estimation model (simple encoder-decoder architecture).
    """
    def __init__(self, input_channels=3):
        super(BaseDepthModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, input_image):
        """
        Forward pass of the base model.
        
        Args:
            input_image: Input image tensor (B, C, H, W)
            
        Returns:
            depth: Predicted depth map (B, 1, H, W)
        """
        features = self.encoder(input_image)
        depth = self.decoder(features)
        return depth


class IDRRGARModel(BaseDepthModel):
    """
    Iterative Dynamic Region Refinement with Geometry-Aware Reconstruction model.
    """
    def __init__(self, input_channels=3, iterations=3):
        super(IDRRGARModel, self).__init__(input_channels)
        self.iterations = iterations
        self.refine_conv = nn.Conv2d(input_channels + 1, 1, kernel_size=3, padding=1)

    def transformer_refine(self, input_image, depth_est):
        """
        Refinement module that uses attention to improve depth estimates.
        
        Args:
            input_image: Input image tensor (B, C, H, W)
            depth_est: Current depth estimate (B, 1, H, W)
            
        Returns:
            delta: Refinement to the current depth estimate (B, 1, H, W)
        """
        B, C, H, W = input_image.shape
        depth_expanded = depth_est.repeat(1, C, 1, 1)
        x = torch.cat([input_image, depth_est], dim=1)  # shape: (B, C+1, H, W)
        delta = self.refine_conv(x)
        return delta

    def forward(self, input_image):
        """
        Forward pass with iterative refinement.
        
        Args:
            input_image: Input image tensor (B, C, H, W)
            
        Returns:
            depth_est: Final depth estimate after refinement (B, 1, H, W)
        """
        depth_est = super(IDRRGARModel, self).forward(input_image)
        
        for i in range(self.iterations):
            delta = self.transformer_refine(input_image, depth_est)
            depth_est = depth_est + delta
            
        return depth_est


def geometry_aware_sampling(feature_map, num_regions=50):
    """
    Sample regions with high gradient magnitude for refinement.
    
    Args:
        feature_map: Feature map tensor (B, C, H, W)
        num_regions: Number of regions to sample
        
    Returns:
        sampled_regions: List of indices for sampled regions
    """
    grad_x = torch.abs(feature_map[:, :, :, :-1] - feature_map[:, :, :, 1:])
    grad_y = torch.abs(feature_map[:, :, :-1, :] - feature_map[:, :, 1:, :])
    
    grad_x = F.pad(grad_x, (0, 1), "constant", 0)
    grad_y = F.pad(grad_y, (0, 0, 0, 1), "constant", 0)
    
    uncertainty = torch.mean(grad_x + grad_y, dim=1)  # shape (B, H, W)
    B, H, W = uncertainty.shape
    
    sampled_regions = []
    for b in range(B):
        uncertainty_flat = uncertainty[b].view(-1)
        _, indices = torch.topk(uncertainty_flat, num_regions)
        sampled_regions.append(indices.cpu().numpy())
        
    return sampled_regions


def uniform_sampling(feature_map, num_regions=50):
    """
    Sample random regions uniformly for comparison with geometry-aware sampling.
    
    Args:
        feature_map: Feature map tensor (B, C, H, W)
        num_regions: Number of regions to sample
        
    Returns:
        sampled_regions: List of indices for sampled regions
    """
    B, C, H, W = feature_map.shape
    total_pixels = H * W
    sampled_regions = []
    
    for b in range(B):
        indices = np.random.choice(total_pixels, num_regions, replace=False)
        sampled_regions.append(indices)
        
    return sampled_regions


def photometric_loss(pred, target):
    """
    L1 loss between predicted and target depth.
    
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        
    Returns:
        loss: L1 loss
    """
    return F.l1_loss(pred, target)


def smoothness_loss(depth_map):
    """
    Smoothness loss to enforce smooth depth estimates.
    
    Args:
        depth_map: Depth map tensor
        
    Returns:
        loss: Smoothness loss
    """
    dx = torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:])
    dy = torch.abs(depth_map[:, :, :-1, :] - depth_map[:, :, 1:, :])
    return (dx.mean() + dy.mean())


def scale_alignment_loss(pred, target):
    """
    Scale alignment loss to address scale ambiguity.
    
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        
    Returns:
        loss: Scale alignment loss
    """
    scale_factor = torch.median(target) / (torch.median(pred) + 1e-6)
    return F.l1_loss(pred * scale_factor, target)


def compute_region_loss(pred, target, samples):
    """
    Compute loss for specific sampled regions.
    
    Args:
        pred: Predicted depth map (B, C, H, W)
        target: Ground truth depth map (B, C, H, W)
        samples: List of sampled region indices
        
    Returns:
        loss: Region-specific loss
    """
    loss_total = 0.0
    B, C, H, W = pred.shape
    
    for b in range(B):
        sample_loss = 0.0
        indices = samples[b]
        
        for ind in indices:
            y = ind // W
            x = ind % W
            sample_loss += F.l1_loss(pred[b, :, y, x], target[b, :, y, x])
            
        loss_total += sample_loss / float(len(indices))
        
    return loss_total / float(B)


def train_model(model, train_loader, optimizer, device, config):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        device: Device to use for training
        config: Configuration parameters
        
    Returns:
        loss_history: List of training losses
    """
    model.train()
    loss_history = []
    
    for batch_idx, (images, gt_depth) in enumerate(train_loader):
        images = images.to(device)
        gt_depth = gt_depth.to(device)
        
        pred_depth = model(images)
        
        loss_photo = photometric_loss(pred_depth, gt_depth)
        loss_smooth = smoothness_loss(pred_depth)
        loss_scale = scale_alignment_loss(pred_depth, gt_depth)
        
        loss = (config.PHOTOMETRIC_LOSS_WEIGHT * loss_photo + 
                config.SMOOTHNESS_LOSS_WEIGHT * loss_smooth + 
                config.SCALE_ALIGNMENT_LOSS_WEIGHT * loss_scale)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if batch_idx % 5 == 0:
            print(f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    return loss_history
