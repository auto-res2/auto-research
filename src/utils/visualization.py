"""
Visualization utilities for the Progressive Brightness Distillation Diffusion experiment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def plot_brightness_histogram(img_tensor, title, filename):
    """
    Plot and save a histogram of brightness values from an image tensor.
    
    Args:
        img_tensor: Tensor image of shape [B, C, H, W]
        title: Title for the plot
        filename: Output filename for saving the plot
    """
    img = img_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
    brightness = img.mean(axis=2).flatten()
    
    plt.figure()
    plt.hist(brightness, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Brightness value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    print(f"Saved histogram plot to {filename}")

def plot_loss(losses, title, filename):
    """
    Plot and save a loss curve.
    
    Args:
        losses: List of loss values
        title: Title for the plot
        filename: Output filename for saving the plot
    """
    plt.figure()
    plt.plot(losses, marker='o')
    plt.title(title)
    plt.xlabel('Epoch/Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    print(f"Saved loss curve to {filename}")

def compute_image_metrics(output_tensor, ground_truth_tensor):
    """
    Compute PSNR and SSIM metrics between output and ground truth images.
    
    Args:
        output_tensor: Output image tensor of shape [B, C, H, W]
        ground_truth_tensor: Ground truth image tensor of shape [B, C, H, W]
        
    Returns:
        tuple: (psnr, ssim) values
    """
    out_np = output_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
    gt_np = ground_truth_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
    
    psnr_val = compute_psnr(gt_np, out_np, data_range=out_np.max()-out_np.min())
    
    min_dim = min(out_np.shape[0], out_np.shape[1])
    win_size = min(7, min_dim - (min_dim % 2) + 1)  # Ensure it's odd and smaller than the image dimensions
    
    data_range = out_np.max() - out_np.min()
    ssim_val = compute_ssim(gt_np, out_np, win_size=win_size, channel_axis=2, data_range=data_range)
    
    return psnr_val, ssim_val
