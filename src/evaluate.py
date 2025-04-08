"""
Evaluation script for ADNLCC method.

This script evaluates the trained diffusion model using various metrics
and visualizes the results with high-quality PDF plots.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
import torchvision.utils as vutils
from tqdm import tqdm

def sample_generation(model, noise_shape, device, steps=50, start_noise_level=0.5):
    """
    Generate samples by iteratively denoising from random noise.
    
    Args:
        model: The trained diffusion model
        noise_shape: Shape of the initial noise tensor [B, C, H, W]
        device: Device to run generation on
        steps: Number of denoising steps
        start_noise_level: Initial noise level to start from
        
    Returns:
        torch.Tensor: Generated samples
    """
    model.eval()
    samples = torch.randn(noise_shape).to(device) * start_noise_level
    
    with torch.no_grad():
        for step in range(steps):
            current_noise_level = start_noise_level * (1 - step / steps)
            
            noise_pred = model(samples, current_noise_level)
            
            samples = samples - noise_pred * (1.0 / steps)
            
            if step < steps - 1:  # Don't add noise at last step
                samples = samples + torch.randn_like(samples) * (current_noise_level / 10)
    
    samples = torch.clamp(samples, -1, 1)
    return samples

def compute_fid(generated_samples, real_samples, device):
    """
    Compute FID score between generated and real samples.
    
    Args:
        generated_samples: Tensor of generated images [B, C, H, W]
        real_samples: Tensor of real images [B, C, H, W]
        device: Device to run FID computation on
        
    Returns:
        float: FID score
    """
    generated_samples = (generated_samples * 0.5 + 0.5).clamp(0, 1)
    real_samples = (real_samples * 0.5 + 0.5).clamp(0, 1)
    
    fid = FrechetInceptionDistance(feature=64).to(device)
    
    fid.update(real_samples, real=True)
    fid.update(generated_samples, real=False)
    
    fid_score = fid.compute().item()
    return fid_score

def compute_ssim_with_nearest(generated_samples, real_dataset):
    """
    For each generated sample, find its nearest neighbor in the training set
    and compute SSIM between them to measure memorization.
    
    Args:
        generated_samples: Tensor of generated samples [B, C, H, W]
        real_dataset: Dataset containing real samples
        
    Returns:
        float: Average SSIM score (higher means more memorization)
    """
    gen_samples = (generated_samples * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    gen_samples_flat = gen_samples.reshape(gen_samples.shape[0], -1)
    
    real_samples_list = []
    for i in range(min(len(real_dataset), 1000)):  # Limit to 1000 real samples for speed
        img, _ = real_dataset[i]
        img_np = (img * 0.5 + 0.5).clamp(0, 1).numpy()
        real_samples_list.append(img_np.flatten())
    real_samples_np = np.array(real_samples_list)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_samples_np)
    
    ssim_scores = []
    for i in range(gen_samples.shape[0]):
        distances, indices = nbrs.kneighbors(gen_samples_flat[i].reshape(1, -1))
        nearest_idx = indices[0][0]
        
        gen_img = gen_samples[i]
        real_img = real_samples_np[nearest_idx].reshape(gen_img.shape)
        
        ssim_channels = []
        for c in range(gen_img.shape[0]):
            ssim_val = ssim(gen_img[c], real_img[c], data_range=1.0)
            ssim_channels.append(ssim_val)
        ssim_scores.append(np.mean(ssim_channels))
    
    return np.mean(ssim_scores), ssim_scores

def visualize_samples(samples, title, filename):
    """
    Visualize and save sample images as high-quality PDF.
    
    Args:
        samples: Tensor of images [B, C, H, W]
        title: Title for the plot
        filename: Path to save the PDF file
        
    Returns:
        str: Path to the saved file
    """
    samples = (samples * 0.5 + 0.5).clamp(0, 1)
    
    grid = vutils.make_grid(samples, nrow=8, padding=2, normalize=False)
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_ssim_histogram(ssim_values_base, ssim_values_adnlcc, filename):
    """
    Plot histograms of SSIM values to compare memorization.
    
    Args:
        ssim_values_base: List of SSIM values for base method
        ssim_values_adnlcc: List of SSIM values for ADNLCC method
        filename: Path to save the PDF file
        
    Returns:
        str: Path to the saved file
    """
    plt.figure(figsize=(10, 6))
    plt.hist(ssim_values_base, bins=20, alpha=0.5, label='Base Method')
    plt.hist(ssim_values_adnlcc, bins=20, alpha=0.5, label='ADNLCC')
    plt.xlabel('SSIM Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of SSIM Scores (Nearest Neighbors)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def evaluate_model(model_base, model_adnlcc, test_loader, real_dataset, device, config):
    """
    Evaluate and compare the base and ADNLCC models.
    
    Args:
        model_base: Base diffusion model
        model_adnlcc: ADNLCC diffusion model
        test_loader: DataLoader for test data
        real_dataset: Dataset containing real samples
        device: Device to run evaluation on
        config: Configuration dictionary
        
    Returns:
        dict: Dictionary of evaluation results
    """
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    real_batch = next(iter(test_loader))[0].to(device)
    batch_size, channels, height, width = real_batch.shape
    
    print("Generating samples with Base model...")
    samples_base = sample_generation(
        model_base, (batch_size, channels, height, width), device, 
        steps=config.get('sampling_steps', 50)
    )
    
    print("Generating samples with ADNLCC model...")
    samples_adnlcc = sample_generation(
        model_adnlcc, (batch_size, channels, height, width), device,
        steps=config.get('sampling_steps', 50)
    )
    
    print("Computing FID scores...")
    fid_base = compute_fid(samples_base, real_batch, device)
    fid_adnlcc = compute_fid(samples_adnlcc, real_batch, device)
    
    print("Computing SSIM for memorization analysis...")
    ssim_base, ssim_values_base = compute_ssim_with_nearest(samples_base, real_dataset)
    ssim_adnlcc, ssim_values_adnlcc = compute_ssim_with_nearest(samples_adnlcc, real_dataset)
    
    print("Visualizing results...")
    base_samples_path = os.path.join(logs_dir, 'generated_samples_base.pdf')
    adnlcc_samples_path = os.path.join(logs_dir, 'generated_samples_adnlcc.pdf')
    ssim_hist_path = os.path.join(logs_dir, 'memorization_ssim_comparison.pdf')
    
    visualize_samples(samples_base, "Generated Samples - Base Method", base_samples_path)
    visualize_samples(samples_adnlcc, "Generated Samples - ADNLCC Method", adnlcc_samples_path)
    
    plot_ssim_histogram(ssim_values_base, ssim_values_adnlcc, ssim_hist_path)
    
    results = {
        'fid_base': fid_base,
        'fid_adnlcc': fid_adnlcc,
        'ssim_base': ssim_base,
        'ssim_adnlcc': ssim_adnlcc,
        'sample_paths': {
            'base': base_samples_path,
            'adnlcc': adnlcc_samples_path,
            'ssim_hist': ssim_hist_path
        }
    }
    
    return results
