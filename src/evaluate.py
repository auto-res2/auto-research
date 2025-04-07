"""
Evaluation module for GCAD experiments.

This module contains functions for evaluating models
and computing various metrics.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
from .preprocess import corrupt_images

def evaluate_model(model, data_loader, device, noise_type="gaussian"):
    """
    Evaluate the model on the validation set with the given corruption scheme,
    computing MSE, PSNR and SSIM (using the first sample in each batch for SSIM).
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for validation data
        device: Device to run the evaluation on
        noise_type: Type of corruption to apply
        
    Returns:
        mse_avg, psnr_avg, ssim_avg
    """
    model.eval()
    mse_total, ssim_total, count = 0.0, 0.0, 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            corrupted = corrupt_images(images, noise_type=noise_type, noise_level=0.1)
            outputs = model(corrupted)
            mse_batch = ((outputs - images)**2).mean().item()
            mse_total += mse_batch
            outputs_np = outputs[0].cpu().permute(1, 2, 0).clamp(0,1).numpy()
            orig_np = images[0].cpu().permute(1, 2, 0).clamp(0,1).numpy()
            ssim_val = ssim(orig_np, outputs_np, win_size=5, channel_axis=2, data_range=1.0)
            ssim_total += ssim_val
            count += 1
    mse_avg = mse_total / count
    psnr_avg = compute_psnr(mse_avg)
    ssim_avg = ssim_total / count
    return mse_avg, psnr_avg, ssim_avg

def evaluate_lpips(model, data_loader, device, noise_type="linear"):
    """
    Evaluate perceptual similarity using LPIPS. Lower scores mean better perceptual similarity.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for validation data
        device: Device to run the evaluation on
        noise_type: Type of corruption to apply
        
    Returns:
        lpips_score_avg
    """
    model.eval()
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_score_total = 0.0
    count = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            corrupted = corrupt_images(images, noise_type=noise_type, noise_level=0.1)
            outputs = model(corrupted)
            for i in range(images.size(0)):
                score = lpips_fn(outputs[i].unsqueeze(0), images[i].unsqueeze(0)).item()
                lpips_score_total += score
                count += 1
    return lpips_score_total / count

def compute_psnr(mse, max_pixel=1.0):
    """Compute Peak Signal-to-Noise Ratio given MSE (using numpy log10)"""
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse)
