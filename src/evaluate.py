import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import time

from utils.utils import load_config, set_seed, get_device, load_model
from utils.networks import TwiSTModel, SIDModel
from utils.metrics import compute_psnr, compute_ssim, add_controlled_noise

def evaluate_model_on_noise_levels(model, test_loader, device, noise_levels, model_name="TwiST"):
    """
    Evaluate model on different noise levels.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        noise_levels: List of noise levels to evaluate on
        model_name: Name of the model for logging
    
    Returns:
        Dictionary of metrics for each noise level
    """
    model.eval()
    metrics = {}
    
    for noise_std in noise_levels:
        print(f"Evaluating {model_name} on noise level {noise_std}...")
        psnr_values = []
        ssim_values = []
        
        with torch.no_grad():
            for data, _ in tqdm(test_loader, desc=f"Noise {noise_std}"):
                data = data.to(device)
                
                # Add noise to input
                data_noisy = add_controlled_noise(data, noise_std)
                
                # Generate output
                if isinstance(model, TwiSTModel):
                    gen, _ = model(data_noisy)
                else:  # SIDModel
                    gen, _, _ = model(data_noisy)
                
                # Compute metrics
                psnr = compute_psnr(gen, data)
                ssim = compute_ssim(gen, data)
                
                psnr_values.append(psnr)
                ssim_values.append(ssim)
        
        # Compute average metrics
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        
        print(f"{model_name} @ noise {noise_std}: PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}")
        
        metrics[noise_std] = {
            "psnr": avg_psnr,
            "ssim": avg_ssim
        }
    
    return metrics

def evaluate_memory_usage(model, test_loader, device, model_name="TwiST"):
    """
    Evaluate memory usage and inference speed.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        model_name: Name of the model for logging
    
    Returns:
        Dictionary of memory usage and inference speed metrics
    """
    model.eval()
    
    # Reset GPU memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    # Warm-up
    data_sample, _ = next(iter(test_loader))
    data_sample = data_sample.to(device)
    
    if isinstance(model, TwiSTModel):
        _ = model(data_sample)
    else:  # SIDModel
        _ = model(data_sample)
    
    # Measure inference time
    start_time = time.time()
    num_samples = 0
    
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc=f"{model_name} Speed Test"):
            data = data.to(device)
            batch_size = data.size(0)
            
            if isinstance(model, TwiSTModel):
                _ = model(data)
            else:  # SIDModel
                _ = model(data)
            
            num_samples += batch_size
    
    end_time = time.time()
    inference_time = end_time - start_time
    samples_per_second = num_samples / inference_time
    
    # Get peak memory usage
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    else:
        peak_memory = "N/A (CPU)"
    
    print(f"{model_name} Memory Usage: {peak_memory} MB")
    print(f"{model_name} Inference Speed: {samples_per_second:.2f} samples/sec")
    
    return {
        "peak_memory_mb": peak_memory,
        "inference_time_sec": inference_time,
        "samples_per_second": samples_per_second,
        "total_samples": num_samples
    }

def compare_models(twist_model, sid_model, test_loader, device, config):
    """
    Compare TwiST-Distill and SID models.
    
    Args:
        twist_model: TwiST-Distill model
        sid_model: SID model
        test_loader: Test data loader
        device: Device to use
        config: Configuration dictionary
    
    Returns:
        Dictionary of comparison metrics
    """
    print("\n" + "="*50)
    print("Comparing TwiST-Distill and SID models")
    print("="*50)
    
    # Evaluate on different noise levels
    noise_levels = config['evaluation']['noise_levels']
    
    twist_metrics = evaluate_model_on_noise_levels(
        twist_model, test_loader, device, noise_levels, "TwiST"
    )
    
    sid_metrics = evaluate_model_on_noise_levels(
        sid_model, test_loader, device, noise_levels, "SID"
    )
    
    # Evaluate memory usage and inference speed
    print("\n" + "="*50)
    print("Memory Usage and Inference Speed")
    print("="*50)
    
    twist_memory = evaluate_memory_usage(twist_model, test_loader, device, "TwiST")
    sid_memory = evaluate_memory_usage(sid_model, test_loader, device, "SID")
    
    # Print comparison summary
    print("\n" + "="*50)
    print("Comparison Summary")
    print("="*50)
    
    print("\nNoise Robustness (PSNR):")
    for noise_std in noise_levels:
        twist_psnr = twist_metrics[noise_std]["psnr"]
        sid_psnr = sid_metrics[noise_std]["psnr"]
        diff = twist_psnr - sid_psnr
        better = "TwiST" if diff > 0 else "SID"
        print(f"  Noise {noise_std}: TwiST = {twist_psnr:.4f}, SID = {sid_psnr:.4f}, Diff = {diff:.4f} ({better} better)")
    
    print("\nNoise Robustness (SSIM):")
    for noise_std in noise_levels:
        twist_ssim = twist_metrics[noise_std]["ssim"]
        sid_ssim = sid_metrics[noise_std]["ssim"]
        diff = twist_ssim - sid_ssim
        better = "TwiST" if diff > 0 else "SID"
        print(f"  Noise {noise_std}: TwiST = {twist_ssim:.4f}, SID = {sid_ssim:.4f}, Diff = {diff:.4f} ({better} better)")
    
    print("\nMemory Usage:")
    if isinstance(twist_memory["peak_memory_mb"], (int, float)) and isinstance(sid_memory["peak_memory_mb"], (int, float)):
        memory_reduction = (sid_memory["peak_memory_mb"] - twist_memory["peak_memory_mb"]) / sid_memory["peak_memory_mb"] * 100
        print(f"  TwiST = {twist_memory['peak_memory_mb']:.2f} MB, SID = {sid_memory['peak_memory_mb']:.2f} MB")
        print(f"  Memory reduction: {memory_reduction:.2f}%")
    else:
        print(f"  TwiST = {twist_memory['peak_memory_mb']}, SID = {sid_memory['peak_memory_mb']}")
    
    print("\nInference Speed:")
    speed_improvement = (twist_memory["samples_per_second"] / sid_memory["samples_per_second"] - 1) * 100
    print(f"  TwiST = {twist_memory['samples_per_second']:.2f} samples/sec, SID = {sid_memory['samples_per_second']:.2f} samples/sec")
    print(f"  Speed improvement: {speed_improvement:.2f}%")
    
    return {
        "twist_metrics": twist_metrics,
        "sid_metrics": sid_metrics,
        "twist_memory": twist_memory,
        "sid_memory": sid_memory
    }

def evaluate_model(config_path, twist_model_path, sid_model_path):
    """
    Evaluate models based on configuration.
    
    Args:
        config_path: Path to configuration file
        twist_model_path: Path to TwiST-Distill model checkpoint
        sid_model_path: Path to SID model checkpoint
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Get device
    device = get_device(config['experiment']['gpu_id'])
    print(f"Using device: {device}")
    
    # Set up models
    if config['model']['architecture'] == 'simple_cnn':
        # Create TwiST-Distill model
        twist_model = TwiSTModel(
            in_channels=config['model']['in_channels'],
            feature_dim=config['model']['feature_dim'],
            hidden_dim=config['model']['hidden_dim'],
            use_bn=config['model']['use_batch_norm']
        ).to(device)
        
        # Create SID model
        sid_model = SIDModel(
            in_channels=config['model']['in_channels'],
            feature_dim=config['model']['feature_dim'],
            hidden_dim=config['model']['hidden_dim'],
            use_bn=config['model']['use_batch_norm']
        ).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {config['model']['architecture']}")
    
    # Load model checkpoints
    if os.path.exists(twist_model_path):
        twist_epoch = load_model(twist_model, twist_model_path, device)
    else:
        print(f"Warning: TwiST model checkpoint not found at {twist_model_path}")
        twist_epoch = 0
    
    if os.path.exists(sid_model_path):
        sid_epoch = load_model(sid_model, sid_model_path, device)
    else:
        print(f"Warning: SID model checkpoint not found at {sid_model_path}")
        sid_epoch = 0
    
    # Get data loaders
    from src.preprocess import preprocess_data
    _, test_loader = preprocess_data(config_path)
    
    # Compare models
    comparison_metrics = compare_models(twist_model, sid_model, test_loader, device, config)
    
    return comparison_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TwiST-Distill model")
    parser.add_argument("--config", type=str, default="config/twist_distill_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--twist_model", type=str, default="models/twist_model_final.pth",
                        help="Path to TwiST-Distill model checkpoint")
    parser.add_argument("--sid_model", type=str, default="models/sid_model_final.pth",
                        help="Path to SID model checkpoint")
    args = parser.parse_args()
    
    metrics = evaluate_model(args.config, args.twist_model, args.sid_model)
    print("Evaluation completed successfully.")
