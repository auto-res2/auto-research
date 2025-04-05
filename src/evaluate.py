"""
ACAG-OVS Evaluation Module

This module contains functions for evaluating models and conducting experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def compute_segmentation_metric(predictions, masks):
    """
    Compute a segmentation metric (IoU-like score).
    
    Args:
        predictions (torch.Tensor): Model predictions
        masks (torch.Tensor): Ground truth masks
        
    Returns:
        float: IoU-like score
    """
    score = torch.sigmoid((predictions - masks).abs().mean())
    return 1.0 - score.item()

def run_experiment1(data_loader, save_dir='logs'):
    """
    Run Experiment 1: Adaptive Attention Calibration Ablation Study.
    
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader
        save_dir (str): Directory to save results
        
    Returns:
        tuple: Average scores for fixed and adaptive thresholding
    """
    from train import AdaptiveThresholding, DummySegmentationModel
    
    print("\nRunning Experiment 1: Adaptive Attention Calibration Ablation Study")
    fixed_module = AdaptiveThresholding(fixed_threshold=0.5)
    adaptive_module = AdaptiveThresholding(k=1.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummySegmentationModel().to(device)
    model.eval()
    
    metric_fixed = 0.0
    metric_adaptive = 0.0
    num_batches = 0

    for images, masks in tqdm(data_loader, desc="Evaluating"):
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            attention_maps, predictions = model(images)
            calibrated_fixed = fixed_module(attention_maps)
            pred_fixed = predictions * (calibrated_fixed.unsqueeze(1))
            score_fixed = compute_segmentation_metric(pred_fixed, masks)
            
            calibrated_adaptive = adaptive_module(attention_maps)
            pred_adaptive = predictions * (calibrated_adaptive.unsqueeze(1))
            score_adaptive = compute_segmentation_metric(pred_adaptive, masks)
            
            metric_fixed += score_fixed
            metric_adaptive += score_adaptive
            num_batches += 1

            print(f"Batch {num_batches}: Fixed score={score_fixed:.4f}, Adaptive score={score_adaptive:.4f}")
    
    avg_fixed = metric_fixed / num_batches
    avg_adaptive = metric_adaptive / num_batches
    print(f"\nExperiment 1 Overall: Avg Fixed Score (IoU-like): {avg_fixed:.4f}, "
          f"Avg Adaptive Score (IoU-like): {avg_adaptive:.4f}")
    
    experiments = ['Fixed Threshold', 'Adaptive Threshold']
    scores = [avg_fixed, avg_adaptive]
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=experiments, y=scores)
    plt.ylabel("Average IoU-like Score")
    plt.title("Experiment 1: Adaptive Attention Calibration")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "adaptive_calibration_comparison.pdf"))
    print(f"Saved plot as '{os.path.join(save_dir, 'adaptive_calibration_comparison.pdf')}'")
    plt.close()
    
    return avg_fixed, avg_adaptive

def run_experiment3(data_loader, save_dir='logs'):
    """
    Run Experiment 3: Architecture-Agnostic Integration.
    
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader
        save_dir (str): Directory to save results
        
    Returns:
        tuple: Scores for different model architectures
    """
    from train import AdaptiveThresholding
    
    print("\nRunning Experiment 3: Architecture-Agnostic Integration")
    
    class DummyBaseModelA(nn.Module):
        def __init__(self, image_channels=3, num_features=8):
            super(DummyBaseModelA, self).__init__()
            self.conv = nn.Conv2d(image_channels, num_features, kernel_size=3, padding=1)
            self.att_conv = nn.Conv2d(num_features, 1, kernel_size=1)

        def forward(self, x, return_attention=False):
            features = F.relu(self.conv(x))
            attention_maps = self.att_conv(features)  # shape (B, 1, H, W)
            if return_attention:
                attention_maps = attention_maps.squeeze(1)
                return attention_maps, features
            return features

    class DummyBaseModelB(nn.Module):
        def __init__(self, image_channels=3, num_features=8):
            super(DummyBaseModelB, self).__init__()
            self.conv = nn.Conv2d(image_channels, num_features, kernel_size=3, padding=1)
            self.att_conv = nn.Conv2d(num_features, 1, kernel_size=1)

        def forward(self, x, return_attention=False):
            features = torch.tanh(self.conv(x))
            attention_maps = self.att_conv(features)
            if return_attention:
                attention_maps = attention_maps.squeeze(1)
                return attention_maps, features
            return features

    class DiffusionModelA(nn.Module):
        def __init__(self, base_model):
            super(DiffusionModelA, self).__init__()
            self.base_model = base_model  # e.g., DummyBaseModelA
            self.adaptive_module = AdaptiveThresholding(k=1.0)
            self.fuse_conv = nn.Conv2d(8, 2, kernel_size=1)  # assuming features have 8 channels

        def forward(self, x):
            attention_maps, features = self.base_model(x, return_attention=True)
            calibrated_attention = self.adaptive_module(attention_maps)  # shape (B, H, W)
            attn_avg = calibrated_attention.unsqueeze(1)  # shape (B,1,H,W)
            fused = features * attn_avg
            predictions = self.fuse_conv(fused)
            return calibrated_attention, predictions

    class DiffusionModelB(nn.Module):
        def __init__(self, base_model):
            super(DiffusionModelB, self).__init__()
            self.base_model = base_model  # e.g., DummyBaseModelB
            self.adaptive_module = AdaptiveThresholding(k=1.0)
            self.fusion_conv = nn.Conv2d(16, 2, kernel_size=1)  # features+attention

        def forward(self, x):
            attention_maps, features = self.base_model(x, return_attention=True)
            calibrated_attention = self.adaptive_module(attention_maps)
            attn_expanded = calibrated_attention.unsqueeze(1).expand_as(features)
            fused = torch.cat([features, attn_expanded], dim=1)
            predictions = self.fusion_conv(fused)
            return calibrated_attention, predictions
    
    def run_architecture_experiment(model_class, base_model, data_loader):
        print(f"\nRunning architecture-agnostic test for model {model_class.__name__}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class(base_model).to(device)
        model.eval()
        metric_accumulator = 0.0
        num_batches = 0
        for images, masks in tqdm(data_loader, desc=f"Testing {model_class.__name__}"):
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                _, predictions = model(images)
                metric_accumulator += compute_segmentation_metric(predictions, masks)
                num_batches += 1
        avg_metric = metric_accumulator / num_batches
        print(f"Average segmentation metric for {model_class.__name__}: {avg_metric:.4f}")
        return avg_metric
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_a = DummyBaseModelA().to(device)
    base_model_b = DummyBaseModelB().to(device)
    
    score_A = run_architecture_experiment(DiffusionModelA, base_model_a, data_loader)
    score_B = run_architecture_experiment(DiffusionModelB, base_model_b, data_loader)
    
    architectures = ['DiffusionModelA', 'DiffusionModelB']
    scores = [score_A, score_B]
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=architectures, y=scores)
    plt.ylabel("Average IoU-like Score")
    plt.title("Experiment 3: Architecture-Agnostic Integration")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "architecture_comparison.pdf"))
    print(f"Saved plot as '{os.path.join(save_dir, 'architecture_comparison.pdf')}'")
    plt.close()
    
    return score_A, score_B
