"""
Evaluation script for the Progressive Brightness Distillation Diffusion experiment.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.pbd_diffusion_config import (
    DEVICE,
    DIFFUSION_STEPS,
    OUTPUT_DIR,
    LOGS_DIR
)
from src.utils.models import InitialBrightnessCorrection, ProgressiveRefinement, TeacherModel
from src.utils.diffusion import reverse_diffusion, run_pipeline
from src.utils.visualization import plot_brightness_histogram, compute_image_metrics

def evaluate_dual_stage_correction(dataloader, gt_dataloader=None):
    """
    Evaluate the dual-stage brightness correction module.
    
    Args:
        dataloader: DataLoader with biased brightness images
        gt_dataloader: DataLoader with ground truth images (optional)
        
    Returns:
        dict: Dictionary with evaluation results
    """
    print("\nStarting Evaluation: Dual-Stage Brightness Correction")
    
    init_module = InitialBrightnessCorrection()
    progressive_module = ProgressiveRefinement()
    teacher_model = TeacherModel()
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    init_module.to(device)
    progressive_module.to(device)
    teacher_model.to(device)
    
    batch = next(iter(dataloader))
    noisy_imgs, _ = batch
    noisy_imgs = noisy_imgs.to(device)
    
    print("Running reverse diffusion simulation (initial + progressive refinement)...")
    outputs = reverse_diffusion(noisy_imgs, init_module, progressive_module, teacher_model, steps=DIFFUSION_STEPS)
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    plot_brightness_histogram(noisy_imgs, "Input (Biased) Brightness Histogram", f"{LOGS_DIR}/brightness_histogram_input.pdf")
    plot_brightness_histogram(outputs[-1], "Corrected Brightness Histogram", f"{LOGS_DIR}/brightness_histogram_corrected.pdf")
    
    results = {
        'input_images': noisy_imgs,
        'corrected_images': outputs[-1],
        'all_outputs': outputs
    }
    
    if gt_dataloader is not None:
        gt_batch = next(iter(gt_dataloader))
        gt_imgs, _ = gt_batch
        gt_imgs = gt_imgs.to(device)
        
        psnr, ssim = compute_image_metrics(outputs[-1], gt_imgs)
        print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
        
        results['gt_images'] = gt_imgs
        results['psnr'] = psnr
        results['ssim'] = ssim
    
    print("Dual-Stage Brightness Correction Evaluation completed.\n")
    return results

def perform_ablation_study(dataloader, gt_dataloader):
    """
    Perform an ablation study on the progressive refinement components.
    
    Args:
        dataloader: DataLoader with biased brightness images
        gt_dataloader: DataLoader with ground truth images
        
    Returns:
        dict: Dictionary with evaluation results for different variants
    """
    print("\nStarting Ablation Study on Progressive Refinement Components")
    
    init_module = InitialBrightnessCorrection()
    progressive_module = ProgressiveRefinement()
    teacher_model = TeacherModel()
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    init_module.to(device)
    progressive_module.to(device)
    teacher_model.to(device)
    
    batch = next(iter(dataloader))
    test_images, _ = batch
    test_images = test_images.to(device)
    
    gt_batch = next(iter(gt_dataloader))
    gt_images, _ = gt_batch
    gt_images = gt_images.to(device)
    
    variants = ['full', 't1_only', 'progressive_only']
    results = {}
    
    for variant in variants:
        print(f"Processing variant: {variant}")
        output = run_pipeline(test_images, init_module, progressive_module, teacher_model, variant)
        results[variant] = output
        
        psnr_val, ssim_val = compute_image_metrics(output, gt_images)
        print(f"Variant: {variant}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        
        pdf_filename = f"{LOGS_DIR}/brightness_histogram_{variant}.pdf"
        plot_brightness_histogram(output, f"{variant} Output Brightness Histogram", pdf_filename)
    
    print("Ablation Study completed.\n")
    return results
