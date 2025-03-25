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
    
    # Print memory usage before evaluation
    if torch.cuda.is_available():
        print(f"GPU Memory before {model_name} evaluation:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
    
    # Process each noise level
    for noise_std in noise_levels:
        print(f"\nEvaluating {model_name} on noise level {noise_std}...")
        psnr_values = []
        ssim_values = []
        batch_times = []
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(test_loader, desc=f"{model_name} @ Noise {noise_std}")):
                batch_start = time.time()
                data = data.to(device)
                
                # Print memory usage periodically
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    print(f"  Batch {batch_idx}: GPU Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
                
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
                
                # Track batch processing time
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Clear intermediate variables to free memory
                del data_noisy, gen
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Compute average metrics
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_batch_time = np.mean(batch_times)
        total_time = time.time() - start_time
        
        print(f"\n{model_name} Results @ noise {noise_std}:")
        print(f"  PSNR = {avg_psnr:.4f} dB")
        print(f"  SSIM = {avg_ssim:.4f}")
        print(f"  Average batch processing time: {avg_batch_time*1000:.2f} ms")
        print(f"  Total evaluation time: {total_time:.2f} seconds")
        
        metrics[noise_std] = {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "avg_batch_time_ms": avg_batch_time*1000,
            "total_time_sec": total_time
        }
        
        # Print memory usage after evaluation
        if torch.cuda.is_available():
            print(f"  GPU Memory after evaluation:")
            print(f"    Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"    Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
    
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
    
    print(f"\n{'-'*20} {model_name} Memory and Speed Evaluation {'-'*20}")
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} Model Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter memory: {total_params * 4 / (1024 ** 2):.2f} MB (assuming float32)")
    
    # Reset GPU memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        print(f"GPU Memory before evaluation:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
    
    # Warm-up
    print(f"Running warm-up inference...")
    data_sample, _ = next(iter(test_loader))
    data_sample = data_sample.to(device)
    
    with torch.no_grad():
        if isinstance(model, TwiSTModel):
            _ = model(data_sample)
        else:  # SIDModel
            _ = model(data_sample)
    
    if device.type == 'cuda':
        print(f"GPU Memory after warm-up:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Peak: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    
    # Measure inference time
    print(f"Measuring inference speed...")
    start_time = time.time()
    num_samples = 0
    batch_times = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(test_loader, desc=f"{model_name} Speed Test")):
            batch_start = time.time()
            data = data.to(device)
            batch_size = data.size(0)
            
            # Print memory usage periodically
            if batch_idx % 20 == 0 and device.type == 'cuda':
                print(f"  Batch {batch_idx}: GPU Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            
            if isinstance(model, TwiSTModel):
                _ = model(data)
            else:  # SIDModel
                _ = model(data)
            
            # Track batch processing time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            num_samples += batch_size
            
            # Clear cache periodically
            if batch_idx % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    end_time = time.time()
    inference_time = end_time - start_time
    samples_per_second = num_samples / inference_time
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    
    # Get peak memory usage
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    else:
        peak_memory = "N/A (CPU)"
        current_memory = "N/A (CPU)"
    
    print(f"\n{model_name} Performance Summary:")
    print(f"  Memory Usage:")
    print(f"    Peak: {peak_memory} MB")
    print(f"    Current: {current_memory} MB")
    print(f"  Inference Speed:")
    print(f"    Total time: {inference_time:.2f} seconds")
    print(f"    Samples per second: {samples_per_second:.2f}")
    print(f"    Average batch time: {avg_batch_time*1000:.2f} ms ± {std_batch_time*1000:.2f} ms")
    print(f"    Total samples processed: {num_samples}")
    
    return {
        "peak_memory_mb": peak_memory,
        "current_memory_mb": current_memory,
        "inference_time_sec": inference_time,
        "samples_per_second": samples_per_second,
        "avg_batch_time_ms": avg_batch_time*1000,
        "std_batch_time_ms": std_batch_time*1000,
        "total_samples": num_samples,
        "total_params": total_params,
        "trainable_params": trainable_params
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
    print("\n" + "="*70)
    print("Comparing TwiST-Distill and SID models".center(70))
    print("="*70)
    
    # Print model architectures
    print("\nModel Architecture Comparison:")
    print(f"  TwiST-Distill: Twin-network with shared parameters")
    print(f"  SID: Three separate networks (f_phi, f_psi, G_theta)")
    
    # Print parameter counts
    twist_params = sum(p.numel() for p in twist_model.parameters())
    sid_params = sum(p.numel() for p in sid_model.parameters())
    param_reduction = (sid_params - twist_params) / sid_params * 100
    
    print("\nParameter Count Comparison:")
    print(f"  TwiST-Distill: {twist_params:,} parameters")
    print(f"  SID: {sid_params:,} parameters")
    print(f"  Parameter reduction: {param_reduction:.2f}%")
    
    # Evaluate on different noise levels
    print("\n" + "="*70)
    print("Noise Robustness Evaluation".center(70))
    print("="*70)
    
    noise_levels = config['evaluation']['noise_levels']
    
    # Process models one at a time to minimize memory usage
    print("\nEvaluating TwiST-Distill model...")
    twist_metrics = evaluate_model_on_noise_levels(
        twist_model, test_loader, device, noise_levels, "TwiST"
    )
    
    # Clear memory before evaluating next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU Memory after TwiST evaluation:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
    
    print("\nEvaluating SID model...")
    sid_metrics = evaluate_model_on_noise_levels(
        sid_model, test_loader, device, noise_levels, "SID"
    )
    
    # Evaluate memory usage and inference speed
    print("\n" + "="*70)
    print("Memory Usage and Inference Speed Evaluation".center(70))
    print("="*70)
    
    # Process models one at a time to minimize memory usage
    twist_memory = evaluate_memory_usage(twist_model, test_loader, device, "TwiST")
    
    # Clear memory before evaluating next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU Memory after TwiST memory evaluation:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
    
    sid_memory = evaluate_memory_usage(sid_model, test_loader, device, "SID")
    
    # Print comprehensive comparison summary
    print("\n" + "="*70)
    print("Comprehensive Comparison Summary".center(70))
    print("="*70)
    
    print("\nNoise Robustness (PSNR):")
    for noise_std in noise_levels:
        twist_psnr = twist_metrics[noise_std]["psnr"]
        sid_psnr = sid_metrics[noise_std]["psnr"]
        diff = twist_psnr - sid_psnr
        better = "TwiST" if diff > 0 else "SID"
        percent_improvement = abs(diff / sid_psnr * 100)
        print(f"  Noise {noise_std}:")
        print(f"    TwiST = {twist_psnr:.4f} dB")
        print(f"    SID = {sid_psnr:.4f} dB")
        print(f"    Difference = {diff:.4f} dB ({better} better by {percent_improvement:.2f}%)")
    
    print("\nNoise Robustness (SSIM):")
    for noise_std in noise_levels:
        twist_ssim = twist_metrics[noise_std]["ssim"]
        sid_ssim = sid_metrics[noise_std]["ssim"]
        diff = twist_ssim - sid_ssim
        better = "TwiST" if diff > 0 else "SID"
        percent_improvement = abs(diff / sid_ssim * 100)
        print(f"  Noise {noise_std}:")
        print(f"    TwiST = {twist_ssim:.4f}")
        print(f"    SID = {sid_ssim:.4f}")
        print(f"    Difference = {diff:.4f} ({better} better by {percent_improvement:.2f}%)")
    
    print("\nMemory Usage:")
    if isinstance(twist_memory["peak_memory_mb"], (int, float)) and isinstance(sid_memory["peak_memory_mb"], (int, float)):
        memory_reduction = (sid_memory["peak_memory_mb"] - twist_memory["peak_memory_mb"]) / sid_memory["peak_memory_mb"] * 100
        print(f"  Peak Memory:")
        print(f"    TwiST = {twist_memory['peak_memory_mb']:.2f} MB")
        print(f"    SID = {sid_memory['peak_memory_mb']:.2f} MB")
        print(f"    Memory reduction: {memory_reduction:.2f}%")
    else:
        print(f"  TwiST = {twist_memory['peak_memory_mb']}, SID = {sid_memory['peak_memory_mb']}")
    
    print("\nInference Speed:")
    speed_improvement = (twist_memory["samples_per_second"] / sid_memory["samples_per_second"] - 1) * 100
    print(f"  Throughput:")
    print(f"    TwiST = {twist_memory['samples_per_second']:.2f} samples/sec")
    print(f"    SID = {sid_memory['samples_per_second']:.2f} samples/sec")
    print(f"    Speed improvement: {speed_improvement:.2f}%")
    
    print(f"\n  Batch Processing Time:")
    print(f"    TwiST = {twist_memory['avg_batch_time_ms']:.2f} ms ± {twist_memory['std_batch_time_ms']:.2f} ms")
    print(f"    SID = {sid_memory['avg_batch_time_ms']:.2f} ms ± {sid_memory['std_batch_time_ms']:.2f} ms")
    
    print("\nParameter Efficiency:")
    params_per_sample_twist = twist_params / twist_memory['samples_per_second']
    params_per_sample_sid = sid_params / sid_memory['samples_per_second']
    efficiency_improvement = (params_per_sample_sid - params_per_sample_twist) / params_per_sample_sid * 100
    print(f"  Parameters per sample per second:")
    print(f"    TwiST = {params_per_sample_twist:.2f}")
    print(f"    SID = {params_per_sample_sid:.2f}")
    print(f"    Efficiency improvement: {efficiency_improvement:.2f}%")
    
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
