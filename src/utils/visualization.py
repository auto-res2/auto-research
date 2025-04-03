"""Visualization utilities for the SPCDD method."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def tensor_to_numpy(tensor):
    """Convert a PyTorch tensor to a numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def plot_image_grid(images, titles=None, rows=1, cols=None, figsize=(10, 10), 
                  save_path=None, cmap='gray'):
    """Plot a grid of images and optionally save as PDF."""
    images = [tensor_to_numpy(img) for img in images]
    
    if cols is None:
        cols = len(images) // rows
        if len(images) % rows != 0:
            cols += 1
            
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap=cmap)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_bar_comparison(values, labels, title="Comparison", 
                      ylabel="Value", save_path=None):
    """Plot a bar chart comparison and save as PDF."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = plt.bar(x, values)
    
    plt.xlabel("Methods")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
