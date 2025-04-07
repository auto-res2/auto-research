"""
Utility functions for evaluating purification methods.
"""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def evaluate_quality(clean, purified):
    """
    Compute PSNR and SSIM metrics between clean and purified images.
    
    Args:
        clean (torch.Tensor): Clean images
        purified (torch.Tensor): Purified images
        
    Returns:
        tuple: (mean PSNR, mean SSIM)
    """
    psnr_vals = []
    ssim_vals = []
    
    for i in range(clean.shape[0]):
        clean_img = clean[i].permute(1, 2, 0).cpu().numpy()
        pur_img = purified[i].permute(1, 2, 0).cpu().numpy()
        
        clean_img = np.clip(clean_img, 0, 1)
        pur_img = np.clip(pur_img, 0, 1)
        
        psnr_vals.append(compute_psnr(clean_img, pur_img, data_range=1))
        ssim_vals.append(compute_ssim(clean_img, pur_img, channel_axis=2, data_range=1, win_size=3))
    
    return np.mean(psnr_vals), np.mean(ssim_vals)

def compute_robust_accuracy(model, images, labels):
    """
    Compute prediction accuracy on the given images.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        images (torch.Tensor): Input images
        labels (torch.Tensor): Ground truth labels
        
    Returns:
        float: Accuracy (0-1)
    """
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
    
    return accuracy
