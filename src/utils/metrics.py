import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
    
    Returns:
        PSNR value as float
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Move channel to last dimension for skimage compatibility
    if img1.shape[1] == 3 or img1.shape[1] == 1:  # [B, C, H, W] format
        img1 = np.transpose(img1, (0, 2, 3, 1))
        img2 = np.transpose(img2, (0, 2, 3, 1))
    
    psnr_values = []
    for i in range(img1.shape[0]):
        psnr_values.append(psnr(img1[i], img2[i], data_range=1.0))
    
    return np.mean(psnr_values)

def compute_ssim(img1, img2):
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
    
    Returns:
        SSIM value as float
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Move channel to last dimension for skimage compatibility
    if img1.shape[1] == 3 or img1.shape[1] == 1:  # [B, C, H, W] format
        img1 = np.transpose(img1, (0, 2, 3, 1))
        img2 = np.transpose(img2, (0, 2, 3, 1))
    
    ssim_values = []
    for i in range(img1.shape[0]):
        ssim_values.append(
            ssim(img1[i], img2[i], data_range=1.0, channel_axis=-1)
        )
    
    return np.mean(ssim_values)

def add_controlled_noise(x, noise_std):
    """
    Add controlled Gaussian noise to tensor.
    
    Args:
        x: Input tensor
        noise_std: Standard deviation of noise
    
    Returns:
        Noisy tensor
    """
    noise = torch.randn_like(x) * noise_std
    return x + noise
