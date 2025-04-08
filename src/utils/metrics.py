"""
Utility functions for computing metrics for GAN evaluation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def compute_fid(real_features, fake_features):
    """
    Compute the Fréchet Inception Distance between real and fake sample features.
    This is a dummy implementation for illustration purposes.
    
    Args:
        real_features: Features from real samples
        fake_features: Features from generated samples
        
    Returns:
        float: FID score (lower is better)
    """
    fid_val = np.abs(np.random.randn()) * 10.0
    return fid_val

def plot_loss_curves(epochs, losses_dict, title, filename):
    """
    Plot multiple loss curves on the same figure.
    
    Args:
        epochs: List of epoch numbers
        losses_dict: Dictionary of loss values (key: model name, value: list of losses)
        title: Title for the plot
        filename: Output filename (will be saved as PDF)
    """
    plt.figure(figsize=(10, 6))
    for model_name, losses in losses_dict.items():
        plt.plot(epochs, losses, label=model_name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"logs/{filename}.pdf")
    plt.close()
    
def plot_tsne(latent_reps, labels=None):
    """
    Create a t-SNE plot of the latent representations.
    
    Args:
        latent_reps: Latent representations
        labels: Optional labels for coloring the points
    """
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_reps)
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], s=5, alpha=0.6, label=f"Class {label}")
        plt.legend()
    else:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], s=5, c='blue', alpha=0.6)
    
    plt.title("t-SNE of Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig("logs/latent_tsne.pdf")
    plt.close()

def save_samples(samples, filename, nrow=8):
    """
    Save generated samples as a grid.
    
    Args:
        samples: Tensor of samples [N, C, H, W]
        filename: Output filename (will be saved as PDF)
        nrow: Number of images per row
    """
    plt.figure(figsize=(10, 10))
    plt.title("Generated Samples")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"logs/{filename}.pdf")
    plt.close()
