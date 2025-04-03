"""Metrics for evaluating model performance."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import matplotlib.pyplot as plt
import os

def calculate_psnr(output, target):
    """Calculate Peak Signal-to-Noise Ratio between output and target."""
    if output.ndim == 4:  # Batch of images
        return np.mean([compute_psnr(o, t) for o, t in zip(output, target)])
    return compute_psnr(output, target)

def calculate_ssim(output, target, data_range=1.0):
    """Calculate Structural Similarity Index between output and target."""
    if output.ndim == 4:  # Batch of images
        return np.mean([compute_ssim(o, t, data_range=data_range) for o, t in zip(output, target)])
    return compute_ssim(output, target, data_range=data_range)

def plot_comparison(output, target, save_path, title="Output vs. Target"):
    """Plot comparison between output and target images and save as PDF."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(output, cmap='gray')
    plt.title("Generated Output")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(target, cmap='gray')
    plt.title("Target")
    plt.axis('off')
    
    plt.suptitle(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_histograms(generated, target, save_path, title="Intensity Distribution"):
    """Plot intensity histograms and save as PDF."""
    plt.figure(figsize=(8, 4))
    plt.hist(generated.flatten(), bins=50, alpha=0.5, label='Generated')
    plt.hist(target.flatten(), bins=50, alpha=0.5, label='Target')
    plt.legend(loc='upper right')
    plt.title(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curve(metrics_dict, save_path, title="Training Progress"):
    """Plot training curves and save as PDF."""
    plt.figure(figsize=(10, 6))
    for label, values in metrics_dict.items():
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
