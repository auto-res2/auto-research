"""Evaluation for CSTD experiments."""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from config import cstd_config as cfg
from src.preprocess import implant_trigger

def evaluate_detection(refiner, data_loader, trigger, mask, device):
    """Evaluate trigger detection performance.
    
    Args:
        refiner: The trained refiner model
        data_loader: DataLoader for evaluation
        trigger: The trigger pattern
        mask: The mask indicating trigger location
        device: Device to run evaluation on
        
    Returns:
        tpr: True Positive Rate
        tnr: True Negative Rate
    """
    refiner.eval()
    tpr_list = []
    tnr_list = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            # Create attacked images
            images_attacked = implant_trigger(images.clone(), trigger, mask, ratio=cfg.TRIGGER_IMPLANT_RATIO)
            est_trigger, _ = refiner(images_attacked)
            
            # For demonstration, use MSE to determine if trigger is detected
            mse = criterion(est_trigger, trigger.expand_as(est_trigger)).item()
            threshold = 0.1  # Arbitrary threshold for demonstration
            
            # Dummy calculation for TPR/TNR
            # In a real implementation, this would be more sophisticated
            detected = mse < threshold
            tpr = 1.0 if detected else 0.0
            tnr = 0.0 if detected else 1.0
            
            tpr_list.append(tpr)
            tnr_list.append(tnr)
            
            # For test, use one iteration
            if cfg.TEST_MODE:
                break
                
    return np.mean(tpr_list), np.mean(tnr_list)

def evaluate_model_performance(model, data_loader, device):
    """Evaluate model inference performance.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        avg_time_ms: Average inference time per image in milliseconds
        memory_mb: Peak memory usage in MB (if CUDA device)
    """
    model.eval()
    total_time = 0.0
    total_samples = 0
    
    # Reset CUDA memory stats if using GPU
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            # Measure inference time
            start_time = time.time()
            _ = model(images)
            total_time += time.time() - start_time
            total_samples += images.size(0)
            
            # For testing, run only one batch
            if cfg.TEST_MODE:
                break
    
    avg_time_ms = (total_time / total_samples) * 1000 if total_samples > 0 else 0
    
    # Get peak memory usage if using GPU
    if device.type == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2  # in MB
    else:
        memory_mb = 0
    
    return avg_time_ms, memory_mb

def compare_models(full_model, distilled_model, data_loader, device):
    """Compare performance between full and distilled models.
    
    Args:
        full_model: The full diffusion model
        distilled_model: The distilled model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
    """
    print("\nComparing Full and Distilled Models:")
    
    # Evaluate full model
    full_time, full_memory = evaluate_model_performance(full_model, data_loader, device)
    print(f"Full Model - Inference Latency: {full_time:.2f} ms/image, Memory: {full_memory:.2f} MB")
    
    # Evaluate distilled model
    distilled_time, distilled_memory = evaluate_model_performance(distilled_model, data_loader, device)
    print(f"Distilled Model - Inference Latency: {distilled_time:.2f} ms/image, Memory: {distilled_memory:.2f} MB")
    
    # Calculate speedup and memory reduction
    speedup = full_time / distilled_time if distilled_time > 0 else 0
    memory_reduction = full_memory / distilled_memory if distilled_memory > 0 else 0
    
    print(f"Speedup: {speedup:.2f}x, Memory Reduction: {memory_reduction:.2f}x")
    
    return {
        'full_time': full_time,
        'distilled_time': distilled_time,
        'full_memory': full_memory,
        'distilled_memory': distilled_memory,
        'speedup': speedup,
        'memory_reduction': memory_reduction
    }
