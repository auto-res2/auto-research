"""
Visualization utilities for the PTDA model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def save_comparison_plot(original_frames, generated_frames, title, filename):
    """
    Save a comparison plot of original and generated frames.
    
    Args:
        original_frames: List of original frames (numpy arrays)
        generated_frames: List of generated frames (numpy arrays)
        title: Title of the plot
        filename: Path to save the plot
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    num_frames = min(len(original_frames), 5)
    
    fig, axes = plt.subplots(2, num_frames, figsize=(4*num_frames, 8))
    
    for i in range(num_frames):
        axes[0, i].imshow(original_frames[i])
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(generated_frames[i])
        axes[1, i].set_title(f"Generated {i+1}")
        axes[1, i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def save_error_map(frame1, frame2, title, filename):
    """
    Save an error map between two frames.
    
    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        title: Title of the plot
        filename: Path to save the plot
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    error_map = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(error_map.astype(np.uint8))
    plt.title(title)
    plt.colorbar(label='Error Magnitude')
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_latent_space(latents, title, filename):
    """
    Visualize the latent space using PCA.
    
    Args:
        latents: List of latent vectors (torch tensors)
        title: Title of the plot
        filename: Path to save the plot
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    latent_array = np.concatenate([l.cpu().numpy() for l in latents], axis=0)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latent_array)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_plot(metrics_dict, title, filename):
    """
    Save a plot of metrics.
    
    Args:
        metrics_dict: Dictionary of metrics (method_name -> list of values)
        title: Title of the plot
        filename: Path to save the plot
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for method, values in metrics_dict.items():
        plt.bar(method, np.mean(values), yerr=np.std(values), capsize=10)
    
    plt.title(title)
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
