#!/usr/bin/env python
"""
TwiST-Distill: Tweedy Score and Consistency Distillation

This script runs the complete experiment for the TwiST-Distill method,
which combines score-based distillation with double-Tweedie consistency loss
to create a memory-efficient and robust one-step generation framework.

The experiment includes:
1. Data preprocessing
2. Model training (both TwiST-Distill and baseline SiD)
3. Model evaluation and comparison
"""

import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.utils import load_config, set_seed, get_device
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

def run_experiment(config_path, test_run=False):
    """
    Run the complete TwiST-Distill experiment.
    
    Args:
        config_path: Path to configuration file
        test_run: Whether to run a quick test with reduced epochs
    """
    # Start timing
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Override test_run if specified
    if test_run:
        config['experiment']['test_run'] = True
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up logging directory
    log_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])
    print(f"Random seed set to {config['experiment']['seed']}")
    
    # Get device
    device = get_device(config['experiment']['gpu_id'])
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.backends.cudnn.version()}")
        # Print detailed GPU information
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        print(f"GPU Memory Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
        print(f"Number of GPU devices: {torch.cuda.device_count()}")
        # Set memory limit if specified
        if 'gpu_memory_limit' in config['experiment'] and config['experiment']['gpu_memory_limit'] > 0:
            torch.cuda.set_per_process_memory_fraction(
                config['experiment']['gpu_memory_limit'] / torch.cuda.get_device_properties(device).total_memory
            )
            print(f"GPU memory limit set to {config['experiment']['gpu_memory_limit']} GB")
    
    # Print experiment configuration
    print("\n" + "="*50)
    print("TwiST-Distill Experiment Configuration")
    print("="*50)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Batch size: {config['dataset']['batch_size']}")
    print(f"Model architecture: {config['model']['architecture']}")
    print(f"Feature dimension: {config['model']['feature_dim']}")
    print(f"Hidden dimension: {config['model']['hidden_dim']}")
    print(f"Training epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Consistency weight: {config['training']['consistency_weight']}")
    print(f"Noise std: {config['training']['noise_std']}")
    print(f"Test run: {config['experiment']['test_run']}")
    print("="*50 + "\n")
    
    # Step 1: Preprocess data
    print("\n" + "="*50)
    print("Step 1: Data Preprocessing")
    print("="*50)
    print(f"Loading dataset: {config['dataset']['name']}")
    print(f"Batch size: {config['dataset']['batch_size']}, Workers: {config['dataset']['num_workers']}")
    train_loader, test_loader = preprocess_data(config_path)
    
    # Step 2: Train models
    print("\n" + "="*50)
    print("Step 2: Model Training")
    print("="*50)
    print(f"Model architecture: {config['model']['architecture']}")
    print(f"Feature dimension: {config['model']['feature_dim']}, Hidden dimension: {config['model']['hidden_dim']}")
    print(f"Training for {config['training']['num_epochs']} epochs (test mode: {config['experiment']['test_run']})")
    twist_model, sid_model, _, _ = train_model(config_path)
    
    # Step 3: Evaluate models
    print("\n" + "="*50)
    print("Step 3: Model Evaluation")
    print("="*50)
    print(f"Evaluating models on {len(config['evaluation']['noise_levels'])} noise levels: {config['evaluation']['noise_levels']}")
    print(f"Metrics to compute: {', '.join(config['evaluation']['metrics'])}")
    twist_model_path = os.path.join(config['training']['save_dir'], "twist_model_final.pth")
    sid_model_path = os.path.join(config['training']['save_dir'], "sid_model_final.pth")
    print(f"TwiST model path: {twist_model_path}")
    print(f"SID model path: {sid_model_path}")
    
    metrics = evaluate_model(config_path, twist_model_path, sid_model_path)
    
    # Calculate total experiment time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print experiment summary
    print("\n" + "="*50)
    print("Experiment Summary")
    print("="*50)
    print(f"Total experiment time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Memory usage comparison
    if isinstance(metrics['twist_memory']['peak_memory_mb'], (int, float)) and isinstance(metrics['sid_memory']['peak_memory_mb'], (int, float)):
        memory_reduction = (metrics['sid_memory']['peak_memory_mb'] - metrics['twist_memory']['peak_memory_mb']) / metrics['sid_memory']['peak_memory_mb'] * 100
        print(f"Memory reduction: {memory_reduction:.2f}%")
    
    # Speed comparison
    speed_improvement = (metrics['twist_memory']['samples_per_second'] / metrics['sid_memory']['samples_per_second'] - 1) * 100
    print(f"Speed improvement: {speed_improvement:.2f}%")
    
    # Noise robustness comparison
    print("\nNoise Robustness Improvement (PSNR):")
    for noise_std in config['evaluation']['noise_levels']:
        twist_psnr = metrics['twist_metrics'][noise_std]["psnr"]
        sid_psnr = metrics['sid_metrics'][noise_std]["psnr"]
        diff = twist_psnr - sid_psnr
        print(f"  Noise {noise_std}: {diff:.4f} dB")
    
    print("\nNoise Robustness Improvement (SSIM):")
    for noise_std in config['evaluation']['noise_levels']:
        twist_ssim = metrics['twist_metrics'][noise_std]["ssim"]
        sid_ssim = metrics['sid_metrics'][noise_std]["ssim"]
        diff = twist_ssim - sid_ssim
        print(f"  Noise {noise_std}: {diff:.4f}")
    
    print("\nExperiment completed successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TwiST-Distill experiment")
    parser.add_argument("--config", type=str, default="config/twist_distill_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--test", action="store_true",
                        help="Run a quick test with reduced epochs")
    args = parser.parse_args()
    
    metrics = run_experiment(args.config, args.test)
