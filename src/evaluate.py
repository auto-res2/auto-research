"""
Evaluation metrics and functions for SAC-Seg experiments.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Intersection over Union (IoU) for segmentation.
    
    Args:
        pred: Predicted segmentation (B, C, H, W) or (B, H, W)
        target: Ground truth segmentation (B, H, W)
        num_classes: Number of segmentation classes
        ignore_index: Index to ignore in calculation
        
    Returns:
        Tuple of (class_ious, mean_iou) tensors
    """
    if pred.dim() > target.dim():
        pred = pred.argmax(1)
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    if ignore_index is not None:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]
    
    class_ious = torch.zeros(num_classes, device=pred.device)
    
    for i in range(num_classes):
        pred_i = pred == i
        target_i = target == i
        intersection = (pred_i & target_i).sum().float()
        union = (pred_i | target_i).sum().float()
        
        if union > 0:
            class_ious[i] = intersection / union
            
    valid_classes = class_ious > 0
    if valid_classes.sum() > 0:
        mean_iou = class_ious[valid_classes].mean()
    else:
        mean_iou = torch.tensor(0.0, device=pred.device)
    
    return class_ious, mean_iou


def compute_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: Optional[int] = None
) -> float:
    """
    Compute pixel accuracy for segmentation.
    
    Args:
        pred: Predicted segmentation (B, C, H, W) or (B, H, W)
        target: Ground truth segmentation (B, H, W)
        ignore_index: Index to ignore in calculation
        
    Returns:
        Pixel accuracy as a float
    """
    if pred.dim() > target.dim():
        pred = pred.argmax(1)
    
    if ignore_index is not None:
        valid = target != ignore_index
        correct = ((pred == target) & valid).sum().float()
        total = valid.sum().float()
    else:
        correct = (pred == target).sum().float()
        total = torch.tensor(target.numel(), device=target.device, dtype=torch.float)
    
    accuracy = correct / total if total > 0 else torch.tensor(0.0, device=target.device)
    return accuracy.item()


def compute_dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Compute Dice coefficient for binary segmentation.
    
    Args:
        pred: Predicted binary segmentation
        target: Ground truth binary segmentation
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient as a float
    """
    pred = (pred > 0).float()
    target = (target > 0).float()
    
    intersection = (pred * target).sum()
    sum_pred_target = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (sum_pred_target + smooth)
    return dice.item()


def segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute multiple segmentation metrics.
    
    Args:
        pred: Predicted segmentation (B, C, H, W)
        target: Ground truth segmentation (B, H, W)
        num_classes: Number of segmentation classes
        ignore_index: Index to ignore in calculation
        
    Returns:
        Dictionary of metrics
    """
    _, mean_iou = compute_iou(pred, target, num_classes, ignore_index)
    pixel_acc = compute_pixel_accuracy(pred, target, ignore_index)
    
    if num_classes == 2:
        if pred.dim() > target.dim():
            pred_binary = pred[:, 1]  # Take the second channel for the foreground
        else:
            pred_binary = pred > 0
            
        target_binary = target > 0
        dice = compute_dice_coefficient(pred_binary, target_binary)
    else:
        dice = 0.0
    
    metrics = {
        'mean_iou': mean_iou.item(),
        'pixel_acc': pixel_acc
    }
    
    if num_classes == 2:
        metrics['dice'] = dice
        
    return metrics
