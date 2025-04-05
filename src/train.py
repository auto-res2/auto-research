"""
ACAG-OVS Training Module

This module contains the model definitions and training functions for the ACAG-OVS experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

class AdaptiveThresholding(nn.Module):
    """
    Adaptive thresholding module for attention calibration.
    """
    def __init__(self, k=1.0, fixed_threshold=None):
        """
        Initialize the adaptive thresholding module.
        
        Args:
            k (float): Hyperparameter for scaling std deviation
            fixed_threshold (float): If provided, the module will use this fixed value instead of adapting
        """
        super(AdaptiveThresholding, self).__init__()
        self.k = k
        self.fixed_threshold = fixed_threshold

    def forward(self, attention_map):
        if self.fixed_threshold is not None:
            threshold = self.fixed_threshold
            if not torch.is_tensor(threshold):
                threshold = torch.tensor(threshold, device=attention_map.device)
            threshold = threshold.expand(attention_map.size(0), 1, 1)
        else:
            B = attention_map.size(0)
            flat_maps = attention_map.view(B, -1)
            mean_vals = flat_maps.mean(dim=1, keepdim=True)
            std_vals = flat_maps.std(dim=1, keepdim=True)
            threshold = mean_vals + self.k * std_vals
            threshold = threshold.view(B, 1, 1)
        calibrated = F.relu(attention_map - threshold)
        return calibrated

class DummySegmentationModel(nn.Module):
    """
    Dummy segmentation model used in Experiment 1.
    """
    def __init__(self, image_channels=3, num_classes=2):
        super(DummySegmentationModel, self).__init__()
        self.conv = nn.Conv2d(image_channels, 8, kernel_size=3, padding=1)
        self.att_conv = nn.Conv2d(8, 1, kernel_size=1)  # output single-channel attention map
        self.out_conv = nn.Conv2d(8, num_classes, kernel_size=1)  # segmentation prediction

    def forward(self, x):
        features = F.relu(self.conv(x))
        attention_maps = self.att_conv(features)
        attention_maps = attention_maps.squeeze(1)
        predictions = self.out_conv(features)
        return attention_maps, predictions

class DummyTokenModel(nn.Module):
    """
    Dummy model with token extraction for Experiment 2.
    """
    def __init__(self, image_channels=3, token_dim=16, num_classes=2):
        super(DummyTokenModel, self).__init__()
        self.conv = nn.Conv2d(image_channels, 8, kernel_size=3, padding=1)
        self.token_conv = nn.Conv2d(8, token_dim, kernel_size=1)
        self.out_conv = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x):
        features = F.relu(self.conv(x))
        predictions = self.out_conv(features)
        tokens = self.token_conv(features)  # tokens are of shape (B, token_dim, H, W)
        tokens_pooled = tokens.mean(dim=[2,3])  # shape (B, token_dim)
        return predictions, tokens_pooled

    def extract_tokens(self, x):
        features = F.relu(self.conv(x))
        tokens = self.token_conv(features)
        tokens_pooled = tokens.mean(dim=[2,3])
        return tokens_pooled

def info_nce_loss(features_1, features_2, temperature=0.07):
    """
    Simple InfoNCE contrastive loss function.
    
    Args:
        features_1, features_2: tensors of shape (B, D)
        temperature (float): Temperature parameter for the loss
        
    Returns:
        torch.Tensor: InfoNCE loss
    """
    B, D = features_1.size()
    features_1 = features_1 / features_1.norm(dim=1, keepdim=True)
    features_2 = features_2 / features_2.norm(dim=1, keepdim=True)
    logits = torch.mm(features_1, features_2.t()) / temperature
    labels = torch.arange(B).to(features_1.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def multi_view_token_extraction(model, images):
    """
    Extract tokens from the model given original and augmented images.
    
    Args:
        model (nn.Module): Token model
        images (torch.Tensor): Tensor of shape (B, 3, H, W)
        
    Returns:
        tuple: Tokens from original view, tokens from augmented view
    """
    tokens_orig = model.extract_tokens(images)
    
    augment = transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])
    
    images_aug = []
    for img in images:
        np_img = img.cpu().numpy().transpose(1, 2, 0)
        img_pil = transforms.ToPILImage()(np_img)
        img_aug = augment(img_pil)
        img_aug_tensor = transforms.ToTensor()(img_aug)
        images_aug.append(img_aug_tensor)
    images_aug = torch.stack(images_aug).to(images.device)
    tokens_aug = model.extract_tokens(images_aug)
    return tokens_orig, tokens_aug

def train_step(model, images, masks, optimizer, use_contrastive=True, alpha=1.0):
    """
    Single training step for Experiment 2.
    
    Args:
        model (nn.Module): Model to train
        images (torch.Tensor): Input images
        masks (torch.Tensor): Target masks
        optimizer (torch.optim.Optimizer): Optimizer
        use_contrastive (bool): Whether to use contrastive loss
        alpha (float): Weight for contrastive loss
        
    Returns:
        tuple: Total loss, segmentation loss
    """
    optimizer.zero_grad()
    predictions, tokens = model(images)
    seg_loss = nn.CrossEntropyLoss()(predictions, masks)
    if use_contrastive:
        tokens_orig, tokens_aug = multi_view_token_extraction(model, images)
        contrast_loss = info_nce_loss(tokens_orig, tokens_aug)
        loss = seg_loss + alpha * contrast_loss
    else:
        loss = seg_loss
    loss.backward()
    optimizer.step()
    return loss.item(), seg_loss.item()
