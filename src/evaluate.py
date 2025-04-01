"""
Evaluation metrics and functions for the SCND experiment.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os

def compute_metrics(output, target):
    """
    Compute SSIM and PSNR between two images.
    
    Args:
        output: Model output tensor (B,C,H,W)
        target: Target tensor (B,C,H,W)
        
    Returns:
        Tuple of (ssim_value, psnr_value)
    """
    output_np = output.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    target_np = target.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    output_gray = np.dot(output_np[...,:3], [0.2989, 0.5870, 0.1140])
    target_gray = np.dot(target_np[...,:3], [0.2989, 0.5870, 0.1140])
    
    ssim_val = compare_ssim(target_gray, output_gray, data_range=target_gray.max()-target_gray.min())
    psnr_val = compare_psnr(target_np, output_np, data_range=target_np.max()-target_np.min())
    return ssim_val, psnr_val

def save_comparison_plot(outputs_list, titles, filename, figsize=(12,6)):
    """
    Create and save a comparison plot of multiple outputs.
    
    Args:
        outputs_list: List of output tensors to visualize
        titles: List of titles for each output
        filename: Output PDF filename
        figsize: Figure size
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig, axs = plt.subplots(1, len(outputs_list), figsize=figsize)
    for i, (output, title) in enumerate(zip(outputs_list, titles)):
        ax = axs[i] if len(outputs_list) > 1 else axs
        ax.set_title(title)
        ax.imshow(output.squeeze(0).permute(1,2,0).detach().cpu().numpy())
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {filename}")
    
def save_loss_curve(loss_values, x_label, y_label, title, filename):
    """
    Create and save a loss curve plot.
    
    Args:
        loss_values: List of loss values
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        filename: Output PDF filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {filename}")
