# src/utils/metrics.py
import torch
import numpy as np
import time

def measure_inference_time(model, latent_shape, num_samples=100, steps=1, device='cuda'):
    """
    Measure the inference time of a model.
    
    Args:
        model: The model to measure
        latent_shape: Shape of input latents
        num_samples: Number of samples to generate
        steps: Number of steps for diffusion models (1 for one-step generator)
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        average_time: Average time per sample in milliseconds
    """
    model.eval()
    model.to(device)
    
    # Warmup
    for _ in range(5):
        latent = torch.randn(*latent_shape, device=device)
        with torch.no_grad():
            if steps > 1:
                # Multi-step inference
                for step in range(steps):
                    latent = model(latent, step)
            else:
                # One-step inference
                _ = model(latent)
    
    # Actual timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_samples):
            latent = torch.randn(*latent_shape, device=device)
            if steps > 1:
                # Multi-step inference
                for step in range(steps):
                    latent = model(latent, step)
            else:
                # One-step inference
                _ = model(latent)
    
    end_time = time.time()
    total_time = end_time - start_time
    average_time = (total_time / num_samples) * 1000  # Convert to milliseconds
    
    return average_time

def compute_memory_usage(model, latent_shape, steps=1, device='cuda'):
    """
    Compute the memory usage of a model during inference.
    
    Args:
        model: The model to measure
        latent_shape: Shape of input latents
        steps: Number of steps for diffusion models (1 for one-step generator)
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        memory_usage: Peak memory usage in MB
    """
    if device != 'cuda':
        return 0  # Memory tracking only works for CUDA
        
    model.eval()
    model.to(device)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run inference
    with torch.no_grad():
        latent = torch.randn(*latent_shape, device=device)
        if steps > 1:
            # Multi-step inference
            for step in range(steps):
                latent = model(latent, step)
        else:
            # One-step inference
            _ = model(latent)
    
    memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    return memory_usage
