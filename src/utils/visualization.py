"""
Visualization utilities for LuminoDiff.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Union


def plot_loss_curve(losses: List[float], 
                   filename: str = "loss_curve.pdf") -> None:
    """
    Plot and save a loss curve.
    
    Args:
        losses (List[float]): List of loss values
        filename (str): Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker="o", linestyle="-")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    print(f"Plot saved as: {filename}")


def plot_attention_map(attention_weights: torch.Tensor, 
                      head_idx: int = 0, 
                      sample_idx: int = 0,
                      filename: str = "attention_map.pdf") -> None:
    """
    Plot and save an attention map.
    
    Args:
        attention_weights (torch.Tensor): Attention weights tensor
        head_idx (int): Index of attention head to visualize
        sample_idx (int): Index of sample to visualize
        filename (str): Filename to save the plot
    """
    if attention_weights is None:
        print("No attention weights to plot")
        return
        
    attn_map = attention_weights[sample_idx, head_idx, :].detach().cpu().numpy()
    map_size = int(np.sqrt(attn_map.shape[-1]))
    attn_map = attn_map.reshape(map_size, map_size)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(attn_map, cmap='viridis')
    plt.colorbar()
    plt.title(f"Attention Map (Head {head_idx})")
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    print(f"Attention map saved as: {filename}")


def plot_image_grid(images: torch.Tensor, 
                   title: str = "Generated Images",
                   filename: str = "image_grid.pdf") -> None:
    """
    Plot and save a grid of images.
    
    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        title (str): Title for the plot
        filename (str): Filename to save the plot
    """
    imgs = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    n = imgs.shape[0]
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    
    plt.figure(figsize=(3*cols, 3*rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        img = np.clip(imgs[i], 0, 1)
        plt.imshow(img)
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    print(f"Image grid saved as: {filename}")
