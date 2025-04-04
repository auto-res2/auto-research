"""
Main script for running CAAD experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocess import (
    ensure_dirs_exist, get_datasets, get_data_loaders,
    get_subloader, add_structured_noise
)
from train import (
    BasicDenoiser, CAADDenoiser, train_model, 
    train_with_validation, save_model, load_model
)
from evaluate import (
    evaluate_denoising, reverse_diffusion, 
    plot_reconstructions, plot_diffusion_convergence, 
    plot_loss_curves
)

sys.path.append('config')
try:
    from caad_config import (
        RANDOM_SEED, DEVICE, DATASET_NAME, BATCH_SIZE, NUM_WORKERS,
        LEARNING_RATE, NUM_EPOCHS, TEST_EPOCHS, NOISE_ALPHA, NOISE_BETA,
        GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, DIFFUSION_ITERATIONS,
        DIFFUSION_STEP_SIZE, DIFFUSION_ERROR_THRESHOLD,
        LIMITED_DATA_PERCENTAGES, FIGURE_DPI
    )
except ImportError:
    print("Warning: Could not import config. Using default values.")
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATASET_NAME = 'CIFAR10'
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    TEST_EPOCHS = 3
    NOISE_ALPHA = 0.8
    NOISE_BETA = 0.2
    GAUSSIAN_KERNEL_SIZE = 7
    GAUSSIAN_SIGMA = 2.0
    DIFFUSION_ITERATIONS = 100
    DIFFUSION_STEP_SIZE = 0.1
    DIFFUSION_ERROR_THRESHOLD = 0.01
    LIMITED_DATA_PERCENTAGES = [0.1, 0.2]
    FIGURE_DPI = 300

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def experiment_structured_noise(test=False):
    """
    Experiment 1: Denoising Quality on Structured Noise.
    
    Args:
        test: if True, run a shortened test version of the experiment
    """
    print("\n--- Experiment 1: Denoising Quality on Structured Noise ---")
    
    device = DEVICE
    print("Using device:", device)
    set_seed(RANDOM_SEED)
    
    train_dataset, test_dataset = get_datasets(DATASET_NAME)
    train_loader, test_loader = get_data_loaders(
        train_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS
    )
    
    def noise_fn(img):
        return add_structured_noise(
            img, NOISE_ALPHA, NOISE_BETA, 
            GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA
        )
    
    base_model = BasicDenoiser().to(device)
    caad_model = CAADDenoiser().to(device)
    
    optimizer_base = optim.Adam(base_model.parameters(), lr=LEARNING_RATE)
    optimizer_caad = optim.Adam(caad_model.parameters(), lr=LEARNING_RATE)
    
    epochs = TEST_EPOCHS if test else NUM_EPOCHS
    
    print("Training Base Method...")
    base_model, _ = train_model(
        base_model, train_loader, optimizer_base, 
        epochs=epochs, device=device, add_noise_fn=noise_fn
    )
    
    print("Training CAAD Method...")
    caad_model, _ = train_model(
        caad_model, train_loader, optimizer_caad, 
        epochs=epochs, device=device, add_noise_fn=noise_fn
    )
    
    save_model(base_model, 'models/base_denoiser.pt')
    save_model(caad_model, 'models/caad_denoiser.pt')
    
    print("Evaluating Base Method...")
    base_psnr, base_ssim, base_samples = evaluate_denoising(
        base_model, test_loader, device, noise_fn
    )
    
    print("Evaluating CAAD Method...")
    caad_psnr, caad_ssim, caad_samples = evaluate_denoising(
        caad_model, test_loader, device, noise_fn
    )
    
    print("\nAverage Metrics:")
    print(f"Base Method: PSNR = {np.mean(base_psnr):.2f}, SSIM = {np.mean(base_ssim):.4f}")
    print(f"CAAD Method: PSNR = {np.mean(caad_psnr):.2f}, SSIM = {np.mean(caad_ssim):.4f}")
    
    plot_reconstructions(
        base_samples['original'][0],
        base_samples['reconstructed'][0],
        caad_samples['reconstructed'][0],
        'reconstructions_pair1.pdf'
    )

def experiment_reverse_diffusion(test=False):
    """
    Experiment 2: Reverse Diffusion Dynamics.
    
    Args:
        test: if True, run a shortened test version of the experiment
    """
    print("\n--- Experiment 2: Reverse Diffusion Dynamics ---")
    
    device = DEVICE
    print("Using device:", device)
    set_seed(RANDOM_SEED)
    
    sample_shape = (1, 3, 32, 32)
    initial_noise = torch.randn(sample_shape).to(device)
    
    try:
        base_model = load_model(BasicDenoiser, 'models/base_denoiser.pt', device)
        caad_model = load_model(CAADDenoiser, 'models/caad_denoiser.pt', device)
        print("Loaded pre-trained models.")
    except:
        print("Could not load pre-trained models. Training new models...")
        train_dataset, test_dataset = get_datasets(DATASET_NAME)
        train_loader, test_loader = get_data_loaders(
            train_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS
        )
        
        def noise_fn(img):
            return add_structured_noise(
                img, NOISE_ALPHA, NOISE_BETA, 
                GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA
            )
        
        base_model = BasicDenoiser().to(device)
        caad_model = CAADDenoiser().to(device)
        
        optimizer_base = optim.Adam(base_model.parameters(), lr=LEARNING_RATE)
        optimizer_caad = optim.Adam(caad_model.parameters(), lr=LEARNING_RATE)
        
        epochs = TEST_EPOCHS if test else NUM_EPOCHS
        
        print("Training Base Method...")
        base_model, _ = train_model(
            base_model, train_loader, optimizer_base, 
            epochs=epochs, device=device, add_noise_fn=noise_fn
        )
        
        print("Training CAAD Method...")
        caad_model, _ = train_model(
            caad_model, train_loader, optimizer_caad, 
            epochs=epochs, device=device, add_noise_fn=noise_fn
        )
        
        save_model(base_model, 'models/base_denoiser.pt')
        save_model(caad_model, 'models/caad_denoiser.pt')
    
    iterations = DIFFUSION_ITERATIONS // 2 if test else DIFFUSION_ITERATIONS
    
    print("Running reverse diffusion for Base Method...")
    refined_base, history_base, iters_base, time_base = reverse_diffusion(
        base_model, initial_noise, iterations=iterations, 
        step_size=DIFFUSION_STEP_SIZE, error_threshold=DIFFUSION_ERROR_THRESHOLD,
        device=device
    )
    
    print("Running reverse diffusion for CAAD Method...")
    refined_caad, history_caad, iters_caad, time_caad = reverse_diffusion(
        caad_model, initial_noise, iterations=iterations, 
        step_size=DIFFUSION_STEP_SIZE, error_threshold=DIFFUSION_ERROR_THRESHOLD,
        device=device
    )
    
    print("\nReverse Diffusion Results:")
    print(f"Base Method: Iterations = {iters_base}, Total Time = {time_base:.4f} sec")
    print(f"CAAD Method: Iterations = {iters_caad}, Total Time = {time_caad:.4f} sec")
    
    plot_diffusion_convergence(history_base, history_caad, 'diffusion_convergence_pair1.pdf')

def experiment_limited_data(test=False):
    """
    Experiment 3: Limited Data Robustness.
    
    Args:
        test: if True, run a shortened test version of the experiment
    """
    print("\n--- Experiment 3: Limited Data Robustness ---")
    
    device = DEVICE
    print("Using device:", device)
    set_seed(RANDOM_SEED)
    
    train_dataset, test_dataset = get_datasets(DATASET_NAME)
    _, test_loader = get_data_loaders(
        train_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS
    )
    
    def noise_fn(img):
        return add_structured_noise(
            img, NOISE_ALPHA, NOISE_BETA, 
            GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA
        )
    
    percentages = [LIMITED_DATA_PERCENTAGES[0]] if test else LIMITED_DATA_PERCENTAGES
    
    for percent in percentages:
        print(f"\nTraining with {percent*100:.0f}% of data:")
        
        train_loader_limited = get_subloader(
            train_dataset, percent, BATCH_SIZE, NUM_WORKERS
        )
        
        base_model = BasicDenoiser().to(device)
        caad_model = CAADDenoiser().to(device)
        
        epochs = TEST_EPOCHS if test else NUM_EPOCHS
        
        print("Training Base Method...")
        base_model, train_losses_base, val_losses_base = train_with_validation(
            base_model, train_loader_limited, test_loader, 
            epochs=epochs, device=device, add_noise_fn=noise_fn
        )
        
        print("Training CAAD Method...")
        caad_model, train_losses_caad, val_losses_caad = train_with_validation(
            caad_model, train_loader_limited, test_loader, 
            epochs=epochs, device=device, add_noise_fn=noise_fn
        )
        
        save_model(base_model, f'models/base_denoiser_{int(percent*100)}pct.pt')
        save_model(caad_model, f'models/caad_denoiser_{int(percent*100)}pct.pt')
        
        plot_loss_curves(
            train_losses_base, val_losses_base, 
            train_losses_caad, val_losses_caad,
            f'loss_{int(percent*100)}pct_pair1.pdf'
        )
        
        print("Evaluating Base Method...")
        base_psnr, base_ssim, _ = evaluate_denoising(
            base_model, test_loader, device, noise_fn
        )
        
        print("Evaluating CAAD Method...")
        caad_psnr, caad_ssim, _ = evaluate_denoising(
            caad_model, test_loader, device, noise_fn
        )
        
        print(f"\nAverage Metrics for {percent*100:.0f}% data:")
        print(f"Base Method: PSNR = {np.mean(base_psnr):.2f}, SSIM = {np.mean(base_ssim):.4f}")
        print(f"CAAD Method: PSNR = {np.mean(caad_psnr):.2f}, SSIM = {np.mean(caad_ssim):.4f}")

def test_code():
    """
    Run a quick test of all experiments.
    This function executes a very brief version of each experiment.
    Execution should finish quickly.
    """
    print("\n=== Running Quick Tests ===")
    
    ensure_dirs_exist()
    
    print("\n[TEST] Experiment 1: Structured Noise and Denoising Quality")
    experiment_structured_noise(test=True)
    
    print("\n[TEST] Experiment 2: Reverse Diffusion Dynamics")
    experiment_reverse_diffusion(test=True)
    
    print("\n[TEST] Experiment 3: Limited Data Robustness")
    experiment_limited_data(test=True)
    
    print("\n=== All quick tests executed successfully! ===")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Run CAAD experiments')
    parser.add_argument('--test', action='store_true', help='Run quick tests')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3], help='Run specific experiment (1, 2, or 3)')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    args = parser.parse_args()
    
    ensure_dirs_exist()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if args.test:
        test_code()
    elif args.exp == 1:
        experiment_structured_noise()
    elif args.exp == 2:
        experiment_reverse_diffusion()
    elif args.exp == 3:
        experiment_limited_data()
    elif args.all:
        experiment_structured_noise()
        experiment_reverse_diffusion()
        experiment_limited_data()
    else:
        test_code()

if __name__ == '__main__':
    main()
