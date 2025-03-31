"""
Model evaluation implementation for LRE-CDT experiment.
This file contains evaluation metrics and visualization functions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_msssim import ssim
import torchvision.utils as vutils

def compute_lpips(gen_img, ref_img):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) metric.
    For demonstration, this uses L1 distance as a proxy.
    
    Args:
        gen_img: Generated image
        ref_img: Reference image
        
    Returns:
        LPIPS value
    """
    return torch.mean(torch.abs(gen_img - ref_img)).item()

def compute_ssim(gen_img, ref_img):
    """
    Compute SSIM (Structural Similarity Index) metric.
    
    Args:
        gen_img: Generated image
        ref_img: Reference image
        
    Returns:
        SSIM value
    """
    gen_img = gen_img.unsqueeze(0)
    ref_img = ref_img.unsqueeze(0)
    return ssim(gen_img, ref_img, data_range=1.0).item()

def compute_dummy_fid(gen_images, ref_images):
    """
    Compute a dummy FID (Fréchet Inception Distance) metric.
    In a real implementation, this would use a pre-trained InceptionV3 model.
    
    Args:
        gen_images: Generated images
        ref_images: Reference images
        
    Returns:
        FID value
    """
    diff = torch.mean(torch.abs(gen_images - ref_images)).item()
    return diff * 100.0

def compute_region_metric(gen_img, ref_img, mask):
    """
    Compute metrics on a specific region defined by a mask.
    
    Args:
        gen_img: Generated image
        ref_img: Reference image
        mask: Region mask
        
    Returns:
        SSIM and LPIPS values for the region
    """
    gen_region = gen_img * mask
    ref_region = ref_img * mask
    ssim_val = compute_ssim(gen_region, ref_region)
    lpips_val = compute_lpips(gen_region, ref_region)
    return ssim_val, lpips_val

def save_comparison_figure(images, outputs, masks, title, filename, save_dir="./logs/"):
    """
    Save a figure comparing input and output images.
    
    Args:
        images: Input images
        outputs: Generated images
        masks: Garment masks
        title: Figure title
        filename: Output filename
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    sample_idx = 0
    sample_image = images[sample_idx].cpu()
    sample_output = outputs[sample_idx].cpu()
    sample_mask = masks[sample_idx].cpu()
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    combined = torch.cat([sample_image, sample_output, sample_mask.expand_as(sample_image)], dim=2)
    ax.imshow(combined.permute(1,2,0))
    ax.axis('off')
    ax.set_title(title)
    plt.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {filepath}")

def save_ablation_figure(image, outputs_dict, masks, title, filename, save_dir="./logs/"):
    """
    Save a figure comparing ablation variants.
    
    Args:
        image: Input image
        outputs_dict: Dictionary of outputs from different model variants
        masks: Garment masks
        title: Figure title
        filename: Output filename
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    sample_idx = 0
    sample_image = image[sample_idx].cpu()
    sample_mask = masks[sample_idx].cpu()
    
    n_variants = len(outputs_dict) + 1  # +1 for input image
    fig, axs = plt.subplots(1, n_variants, figsize=(4*n_variants, 4))
    
    axs[0].imshow(sample_image.permute(1,2,0))
    axs[0].set_title("Input")
    axs[0].axis('off')
    
    for i, (variant_name, output) in enumerate(outputs_dict.items(), 1):
        sample_output = output[sample_idx].cpu()
        axs[i].imshow(sample_output.permute(1,2,0))
        axs[i].set_title(variant_name)
        axs[i].axis('off')
    
    plt.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation figure to {filepath}")

def save_efficiency_plots(step_counts, time_results, quality_results, filename, save_dir="./logs/"):
    """
    Save efficiency tradeoff plots.
    
    Args:
        step_counts: List of diffusion step counts
        time_results: Dictionary of inference times for each model
        quality_results: Dictionary of quality metrics for each model
        filename: Output filename
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    
    for model_name, times in time_results.items():
        axs[0].plot(step_counts, times, marker='o', label=model_name)
    
    axs[0].set_xlabel("Number of Inference Steps")
    axs[0].set_ylabel("Average Inference Time (s)")
    axs[0].set_title("Inference Time vs Diffusion Steps")
    axs[0].legend()
    
    for model_name, metrics in quality_results.items():
        axs[1].plot(step_counts, metrics, marker='o', label=f"{model_name} LPIPS")
    
    axs[1].set_xlabel("Number of Inference Steps")
    axs[1].set_ylabel("LPIPS")
    axs[1].set_title("Quality vs Diffusion Steps (LPIPS)")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved efficiency plots to {filepath}")
