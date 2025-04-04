"""
Model evaluation for CAAD experiments.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import os

def evaluate_denoising(model, test_loader, device='cuda', add_noise_fn=None, num_samples=10):
    """
    Evaluate denoising quality on a test dataset.
    
    Args:
        model: trained denoiser model
        test_loader: data loader for test data
        device: device to use for evaluation
        add_noise_fn: function to add noise to input images
        num_samples: number of samples to evaluate
    
    Returns:
        tuple: (psnr_values, ssim_values, sample_images)
    """
    model.eval()
    psnr_values = []
    ssim_values = []
    sample_images = {'original': [], 'corrupted': [], 'reconstructed': []}
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
                
            data = data.to(device)
            
            if add_noise_fn:
                corrupted = torch.stack([add_noise_fn(img) for img in data]).to(device)
            else:
                corrupted = data
            
            reconstructed = model(corrupted)
            
            for i in range(min(1, len(data))):
                original = data[i].cpu().numpy().transpose(1, 2, 0)
                corrupted_np = corrupted[i].cpu().numpy().transpose(1, 2, 0)
                recon = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
                
                psnr = compute_psnr(original, recon)
                ssim = compute_ssim(original, recon, multichannel=True)
                
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                
                sample_images['original'].append(original)
                sample_images['corrupted'].append(corrupted_np)
                sample_images['reconstructed'].append(recon)
    
    return psnr_values, ssim_values, sample_images

def reverse_diffusion(model, initial_noise, iterations=100, step_size=0.1, 
                      error_threshold=0.01, device='cuda'):
    """
    Perform reverse diffusion process.
    
    Args:
        model: denoiser model
        initial_noise: initial noise tensor
        iterations: maximum number of iterations
        step_size: step size for updates
        error_threshold: error threshold for early stopping
        device: device to use for computation
    
    Returns:
        tuple: (final sample, error history, iterations used, total time)
    """
    model.eval()
    current_sample = initial_noise.clone().to(device)
    history = []
    start_time = time.time()
    criterion = nn.MSELoss()
    
    target = torch.zeros_like(initial_noise).to(device)
    
    for i in range(iterations):
        with torch.no_grad():
            predicted = model(current_sample)
            current_sample = current_sample - step_size * (current_sample - predicted)
            error = criterion(current_sample, target).item()
            history.append(error)
            
            if error < error_threshold:
                break
    
    total_time = time.time() - start_time
    return current_sample, history, i+1, total_time

def plot_reconstructions(original, base_recon, caad_recon, filename='reconstructions_pair1.pdf'):
    """
    Plot and save the sample reconstructions.
    
    Args:
        original: original image
        base_recon: reconstruction from base model
        caad_recon: reconstruction from CAAD model
        filename: output filename
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1,3,1)
    plt.title("Ground Truth")
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.title("Base Reconstruction")
    plt.imshow(base_recon)
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.title("CAAD Reconstruction")
    plt.imshow(caad_recon)
    plt.axis('off')
    
    plt.tight_layout()
    os.makedirs('logs', exist_ok=True)
    plt.savefig(f"logs/{filename}", format='pdf', dpi=300)
    print(f"Reconstruction plot saved as 'logs/{filename}'")
    plt.close()

def plot_diffusion_convergence(base_history, caad_history, filename='diffusion_convergence_pair1.pdf'):
    """
    Plot error evolution for the reverse diffusion iterations.
    
    Args:
        base_history: error history for base model
        caad_history: error history for CAAD model
        filename: output filename
    """
    plt.figure()
    plt.plot(base_history, label='Base')
    plt.plot(caad_history, label='CAAD')
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction Error")
    plt.title("Reverse Diffusion Convergence")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('logs', exist_ok=True)
    plt.savefig(f"logs/{filename}", format='pdf', dpi=300)
    print(f"Reverse diffusion convergence plot saved as 'logs/{filename}'")
    plt.close()

def plot_loss_curves(train_losses_base, val_losses_base, train_losses_caad, val_losses_caad, 
                    filename='loss_10pct_pair1.pdf'):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses_base: training losses for base model
        val_losses_base: validation losses for base model
        train_losses_caad: training losses for CAAD model
        val_losses_caad: validation losses for CAAD model
        filename: output filename
    """
    plt.figure()
    plt.plot(train_losses_base, label="Base Train")
    plt.plot(val_losses_base, label="Base Val")
    plt.plot(train_losses_caad, label="CAAD Train")
    plt.plot(val_losses_caad, label="CAAD Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('logs', exist_ok=True)
    plt.savefig(f"logs/{filename}", format='pdf', dpi=300)
    print(f"Loss curves saved as 'logs/{filename}'")
    plt.close()
