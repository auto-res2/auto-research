"""
Utility functions for diffusion processes in SBDT method.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def forward_diffusion(label_tensor, timesteps=10, beta_start=0.0001, beta_end=0.02):
    """
    Forward diffusion process: adds Gaussian noise over a fixed number of timesteps.
    
    Args:
        label_tensor: Input tensor to apply diffusion
        timesteps: Number of diffusion steps
        beta_start: Initial noise level
        beta_end: Final noise level
        
    Returns:
        List of noisy tensors at each timestep
    """
    betas = torch.linspace(beta_start, beta_end, timesteps)
    noisy_labels = []
    x = label_tensor.clone()
    for beta in betas:
        noise = torch.randn_like(x) * np.sqrt(beta.item())
        x = x + noise
        noisy_labels.append(x)
    return noisy_labels

def evaluate_reconstruction(original, reconstruction):
    """
    Evaluate reconstruction quality using SSIM.
    
    Args:
        original: Original image tensor
        reconstruction: Reconstructed image tensor
        
    Returns:
        SSIM score
    """
    # Convert tensors to numpy arrays
    orig = original.cpu().numpy().transpose(1, 2, 0)
    recon = reconstruction.cpu().numpy().transpose(1, 2, 0)
    
    # Clip values to [0,1] for SSIM computation
    orig = np.clip(orig, 0, 1)
    recon = np.clip(recon, 0, 1)
    
    # Compute SSIM score with a smaller window size for small images
    # CIFAR-10 images are 32x32, so we need a smaller window size
    win_size = min(7, min(orig.shape[0], orig.shape[1]) - 1)
    # Ensure win_size is odd
    if win_size % 2 == 0:
        win_size -= 1
    
    # Use channel_axis instead of multichannel parameter (which is deprecated)
    # Add data_range parameter for floating point images
    score = ssim(orig, recon, win_size=win_size, channel_axis=2, data_range=1.0)
    return score

def plot_reconstructions(originals, reconstructions, title, save_path=None):
    """
    Plot original images and their reconstructions.
    
    Args:
        originals: Batch of original images
        reconstructions: Batch of reconstructed images
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        None
    """
    n = min(4, originals.size(0))
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    
    for i in range(n):
        # Original image
        img = originals[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")
        
        # Reconstructed image
        recon = reconstructions[i].cpu().detach().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(np.clip(recon, 0, 1))
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstruction")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()
