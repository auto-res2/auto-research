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
    print("\n" + "="*80)
    print("EXPERIMENT 1: DENOISING QUALITY ON STRUCTURED NOISE")
    print("="*80)
    
    device = DEVICE
    print(f"Configuration:")
    print(f"  - Device: {device}")
    print(f"  - Random Seed: {RANDOM_SEED}")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Noise Parameters: alpha={NOISE_ALPHA}, beta={NOISE_BETA}")
    print(f"  - Gaussian Filter: kernel_size={GAUSSIAN_KERNEL_SIZE}, sigma={GAUSSIAN_SIGMA}")
    
    set_seed(RANDOM_SEED)
    
    print("\nLoading datasets...")
    train_dataset, test_dataset = get_datasets(DATASET_NAME)
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    
    train_loader, test_loader = get_data_loaders(
        train_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS
    )
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    def noise_fn(img):
        return add_structured_noise(
            img, NOISE_ALPHA, NOISE_BETA, 
            GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA
        )
    
    print("\nInitializing models...")
    base_model = BasicDenoiser().to(device)
    caad_model = CAADDenoiser().to(device)
    
    base_params = sum(p.numel() for p in base_model.parameters())
    caad_params = sum(p.numel() for p in caad_model.parameters())
    print(f"  - Base Model Parameters: {base_params:,}")
    print(f"  - CAAD Model Parameters: {caad_params:,}")
    
    optimizer_base = optim.Adam(base_model.parameters(), lr=LEARNING_RATE)
    optimizer_caad = optim.Adam(caad_model.parameters(), lr=LEARNING_RATE)
    
    epochs = TEST_EPOCHS if test else NUM_EPOCHS
    print(f"\nTraining for {epochs} epochs {'(test mode)' if test else ''}")
    
    print("\n[1/4] Training Base Method...")
    start_time = time.time()
    base_model, base_losses = train_model(
        base_model, train_loader, optimizer_base, 
        epochs=epochs, device=device, add_noise_fn=noise_fn
    )
    base_train_time = time.time() - start_time
    print(f"  - Training completed in {base_train_time:.2f} seconds")
    print(f"  - Final training loss: {base_losses[-1]:.6f}")
    
    print("\n[2/4] Training CAAD Method...")
    start_time = time.time()
    caad_model, caad_losses = train_model(
        caad_model, train_loader, optimizer_caad, 
        epochs=epochs, device=device, add_noise_fn=noise_fn
    )
    caad_train_time = time.time() - start_time
    print(f"  - Training completed in {caad_train_time:.2f} seconds")
    print(f"  - Final training loss: {caad_losses[-1]:.6f}")
    
    print("\nSaving models...")
    save_model(base_model, 'models/base_denoiser.pt')
    save_model(caad_model, 'models/caad_denoiser.pt')
    
    print("\n[3/4] Evaluating Base Method...")
    base_psnr, base_ssim, base_samples = evaluate_denoising(
        base_model, test_loader, device, noise_fn
    )
    print(f"  - Samples evaluated: {len(base_psnr)}")
    print(f"  - PSNR range: [{min(base_psnr):.2f}, {max(base_psnr):.2f}]")
    print(f"  - SSIM range: [{min(base_ssim):.4f}, {max(base_ssim):.4f}]")
    
    print("\n[4/4] Evaluating CAAD Method...")
    caad_psnr, caad_ssim, caad_samples = evaluate_denoising(
        caad_model, test_loader, device, noise_fn
    )
    print(f"  - Samples evaluated: {len(caad_psnr)}")
    print(f"  - PSNR range: [{min(caad_psnr):.2f}, {max(caad_psnr):.2f}]")
    print(f"  - SSIM range: [{min(caad_ssim):.4f}, {max(caad_ssim):.4f}]")
    
    print("\nFinal Results:")
    print(f"  - Base Method: PSNR = {np.mean(base_psnr):.2f} ± {np.std(base_psnr):.2f}, SSIM = {np.mean(base_ssim):.4f} ± {np.std(base_ssim):.4f}")
    print(f"  - CAAD Method: PSNR = {np.mean(caad_psnr):.2f} ± {np.std(caad_psnr):.2f}, SSIM = {np.mean(caad_ssim):.4f} ± {np.std(caad_ssim):.4f}")
    print(f"  - Improvement: PSNR +{np.mean(caad_psnr) - np.mean(base_psnr):.2f}, SSIM +{np.mean(caad_ssim) - np.mean(base_ssim):.4f}")
    
    print("\nGenerating visualization...")
    plot_reconstructions(
        base_samples['original'][0],
        base_samples['reconstructed'][0],
        caad_samples['reconstructed'][0],
        'reconstructions_pair1.pdf'
    )
    print("  - Saved reconstruction comparison to logs/reconstructions_pair1.pdf")

def experiment_reverse_diffusion(test=False):
    """
    Experiment 2: Reverse Diffusion Dynamics.
    
    Args:
        test: if True, run a shortened test version of the experiment
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: REVERSE DIFFUSION DYNAMICS")
    print("="*80)
    
    device = DEVICE
    print(f"Configuration:")
    print(f"  - Device: {device}")
    print(f"  - Random Seed: {RANDOM_SEED}")
    print(f"  - Diffusion Parameters:")
    print(f"    * Iterations: {DIFFUSION_ITERATIONS // 2 if test else DIFFUSION_ITERATIONS} {'(test mode)' if test else ''}")
    print(f"    * Step Size: {DIFFUSION_STEP_SIZE}")
    print(f"    * Error Threshold: {DIFFUSION_ERROR_THRESHOLD}")
    
    set_seed(RANDOM_SEED)
    
    print("\nInitializing noise sample...")
    sample_shape = (1, 3, 32, 32)
    initial_noise = torch.randn(sample_shape).to(device)
    print(f"  - Sample shape: {sample_shape}")
    print(f"  - Initial noise statistics: mean={initial_noise.mean().item():.4f}, std={initial_noise.std().item():.4f}")
    
    print("\nLoading models...")
    models_loaded = False
    try:
        start_time = time.time()
        base_model = load_model(BasicDenoiser, 'models/base_denoiser.pt', device)
        caad_model = load_model(CAADDenoiser, 'models/caad_denoiser.pt', device)
        load_time = time.time() - start_time
        print(f"  - Successfully loaded pre-trained models in {load_time:.2f} seconds")
        models_loaded = True
    except Exception as e:
        print(f"  - Could not load pre-trained models: {str(e)}")
        print("\nTraining new models from scratch...")
        
        print("  - Loading datasets...")
        train_dataset, test_dataset = get_datasets(DATASET_NAME)
        print(f"    * Training samples: {len(train_dataset)}")
        print(f"    * Test samples: {len(test_dataset)}")
        
        train_loader, test_loader = get_data_loaders(
            train_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS
        )
        
        def noise_fn(img):
            return add_structured_noise(
                img, NOISE_ALPHA, NOISE_BETA, 
                GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA
            )
        
        print("  - Initializing models...")
        base_model = BasicDenoiser().to(device)
        caad_model = CAADDenoiser().to(device)
        
        base_params = sum(p.numel() for p in base_model.parameters())
        caad_params = sum(p.numel() for p in caad_model.parameters())
        print(f"    * Base Model Parameters: {base_params:,}")
        print(f"    * CAAD Model Parameters: {caad_params:,}")
        
        optimizer_base = optim.Adam(base_model.parameters(), lr=LEARNING_RATE)
        optimizer_caad = optim.Adam(caad_model.parameters(), lr=LEARNING_RATE)
        
        epochs = TEST_EPOCHS if test else NUM_EPOCHS
        print(f"\n  - Training for {epochs} epochs {'(test mode)' if test else ''}")
        
        print("\n  - Training Base Method...")
        start_time = time.time()
        base_model, base_losses = train_model(
            base_model, train_loader, optimizer_base, 
            epochs=epochs, device=device, add_noise_fn=noise_fn
        )
        base_train_time = time.time() - start_time
        print(f"    * Training completed in {base_train_time:.2f} seconds")
        print(f"    * Final training loss: {base_losses[-1]:.6f}")
        
        print("\n  - Training CAAD Method...")
        start_time = time.time()
        caad_model, caad_losses = train_model(
            caad_model, train_loader, optimizer_caad, 
            epochs=epochs, device=device, add_noise_fn=noise_fn
        )
        caad_train_time = time.time() - start_time
        print(f"    * Training completed in {caad_train_time:.2f} seconds")
        print(f"    * Final training loss: {caad_losses[-1]:.6f}")
        
        print("\n  - Saving models...")
        save_model(base_model, 'models/base_denoiser.pt')
        save_model(caad_model, 'models/caad_denoiser.pt')
    
    iterations = DIFFUSION_ITERATIONS // 2 if test else DIFFUSION_ITERATIONS
    
    print("\n[1/2] Running reverse diffusion for Base Method...")
    start_time = time.time()
    refined_base, history_base, iters_base, time_base = reverse_diffusion(
        base_model, initial_noise, iterations=iterations, 
        step_size=DIFFUSION_STEP_SIZE, error_threshold=DIFFUSION_ERROR_THRESHOLD,
        device=device
    )
    total_time = time.time() - start_time
    print(f"  - Process completed in {total_time:.2f} seconds")
    print(f"  - Iterations used: {iters_base}/{iterations} ({iters_base/iterations*100:.1f}%)")
    print(f"  - Average time per iteration: {time_base/iters_base*1000:.2f} ms")
    print(f"  - Initial error: {history_base[0]:.6f}")
    print(f"  - Final error: {history_base[-1]:.6f}")
    print(f"  - Error reduction: {(1 - history_base[-1]/history_base[0])*100:.2f}%")
    
    print("\n[2/2] Running reverse diffusion for CAAD Method...")
    start_time = time.time()
    refined_caad, history_caad, iters_caad, time_caad = reverse_diffusion(
        caad_model, initial_noise, iterations=iterations, 
        step_size=DIFFUSION_STEP_SIZE, error_threshold=DIFFUSION_ERROR_THRESHOLD,
        device=device
    )
    total_time = time.time() - start_time
    print(f"  - Process completed in {total_time:.2f} seconds")
    print(f"  - Iterations used: {iters_caad}/{iterations} ({iters_caad/iterations*100:.1f}%)")
    print(f"  - Average time per iteration: {time_caad/iters_caad*1000:.2f} ms")
    print(f"  - Initial error: {history_caad[0]:.6f}")
    print(f"  - Final error: {history_caad[-1]:.6f}")
    print(f"  - Error reduction: {(1 - history_caad[-1]/history_caad[0])*100:.2f}%")
    
    print("\nComparative Results:")
    print(f"  - Base Method: {iters_base} iterations, {time_base:.4f} sec total, {time_base/iters_base*1000:.2f} ms/iter")
    print(f"  - CAAD Method: {iters_caad} iterations, {time_caad:.4f} sec total, {time_caad/iters_caad*1000:.2f} ms/iter")
    
    iter_improvement = (iters_base - iters_caad) / iters_base * 100
    time_improvement = (time_base - time_caad) / time_base * 100
    error_improvement = (history_base[-1] - history_caad[-1]) / history_base[-1] * 100
    
    print(f"  - Efficiency Gains:")
    print(f"    * Iteration reduction: {iter_improvement:.1f}%")
    print(f"    * Time reduction: {time_improvement:.1f}%")
    print(f"    * Final error improvement: {error_improvement:.1f}%")
    
    print("\nGenerating visualization...")
    plot_diffusion_convergence(history_base, history_caad, 'diffusion_convergence_pair1.pdf')
    print("  - Saved convergence plot to logs/diffusion_convergence_pair1.pdf")

def experiment_limited_data(test=False):
    """
    Experiment 3: Limited Data Robustness.
    
    Args:
        test: if True, run a shortened test version of the experiment
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: LIMITED DATA ROBUSTNESS")
    print("="*80)
    
    device = DEVICE
    print(f"Configuration:")
    print(f"  - Device: {device}")
    print(f"  - Random Seed: {RANDOM_SEED}")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    
    set_seed(RANDOM_SEED)
    
    print("\nLoading datasets...")
    train_dataset, test_dataset = get_datasets(DATASET_NAME)
    print(f"  - Full training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    
    _, test_loader = get_data_loaders(
        train_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS
    )
    print(f"  - Test batches: {len(test_loader)}")
    
    def noise_fn(img):
        return add_structured_noise(
            img, NOISE_ALPHA, NOISE_BETA, 
            GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA
        )
    
    percentages = [LIMITED_DATA_PERCENTAGES[0]] if test else LIMITED_DATA_PERCENTAGES
    print(f"\nRunning experiments with data percentages: {[p*100 for p in percentages]}% {'(test mode)' if test else ''}")
    
    all_results = []
    
    for percent in percentages:
        print(f"\n{'-'*60}")
        print(f"TRAINING WITH {percent*100:.0f}% OF DATA ({int(len(train_dataset) * percent):,} samples)")
        print(f"{'-'*60}")
        
        train_loader_limited = get_subloader(
            train_dataset, percent, BATCH_SIZE, NUM_WORKERS
        )
        print(f"  - Limited training batches: {len(train_loader_limited)}")
        
        print("\nInitializing models...")
        base_model = BasicDenoiser().to(device)
        caad_model = CAADDenoiser().to(device)
        
        base_params = sum(p.numel() for p in base_model.parameters())
        caad_params = sum(p.numel() for p in caad_model.parameters())
        print(f"  - Base Model Parameters: {base_params:,}")
        print(f"  - CAAD Model Parameters: {caad_params:,}")
        
        epochs = TEST_EPOCHS if test else NUM_EPOCHS
        print(f"\nTraining for {epochs} epochs {'(test mode)' if test else ''}")
        
        print("\n[1/4] Training Base Method with validation...")
        start_time = time.time()
        base_model, train_losses_base, val_losses_base = train_with_validation(
            base_model, train_loader_limited, test_loader, 
            epochs=epochs, device=device, add_noise_fn=noise_fn
        )
        base_train_time = time.time() - start_time
        print(f"  - Training completed in {base_train_time:.2f} seconds")
        print(f"  - Final training loss: {train_losses_base[-1]:.6f}")
        print(f"  - Final validation loss: {val_losses_base[-1]:.6f}")
        print(f"  - Training/validation gap: {(train_losses_base[-1] - val_losses_base[-1])/train_losses_base[-1]*100:.2f}%")
        
        print("\n[2/4] Training CAAD Method with validation...")
        start_time = time.time()
        caad_model, train_losses_caad, val_losses_caad = train_with_validation(
            caad_model, train_loader_limited, test_loader, 
            epochs=epochs, device=device, add_noise_fn=noise_fn
        )
        caad_train_time = time.time() - start_time
        print(f"  - Training completed in {caad_train_time:.2f} seconds")
        print(f"  - Final training loss: {train_losses_caad[-1]:.6f}")
        print(f"  - Final validation loss: {val_losses_caad[-1]:.6f}")
        print(f"  - Training/validation gap: {(train_losses_caad[-1] - val_losses_caad[-1])/train_losses_caad[-1]*100:.2f}%")
        
        print("\nSaving models...")
        save_model(base_model, f'models/base_denoiser_{int(percent*100)}pct.pt')
        save_model(caad_model, f'models/caad_denoiser_{int(percent*100)}pct.pt')
        
        print("\nGenerating loss curves visualization...")
        plot_loss_curves(
            train_losses_base, val_losses_base, 
            train_losses_caad, val_losses_caad,
            f'loss_{int(percent*100)}pct_pair1.pdf'
        )
        print(f"  - Saved loss curves to logs/loss_{int(percent*100)}pct_pair1.pdf")
        
        print("\n[3/4] Evaluating Base Method...")
        start_time = time.time()
        base_psnr, base_ssim, _ = evaluate_denoising(
            base_model, test_loader, device, noise_fn
        )
        base_eval_time = time.time() - start_time
        print(f"  - Evaluation completed in {base_eval_time:.2f} seconds")
        print(f"  - Samples evaluated: {len(base_psnr)}")
        print(f"  - PSNR range: [{min(base_psnr):.2f}, {max(base_psnr):.2f}]")
        print(f"  - SSIM range: [{min(base_ssim):.4f}, {max(base_ssim):.4f}]")
        
        print("\n[4/4] Evaluating CAAD Method...")
        start_time = time.time()
        caad_psnr, caad_ssim, _ = evaluate_denoising(
            caad_model, test_loader, device, noise_fn
        )
        caad_eval_time = time.time() - start_time
        print(f"  - Evaluation completed in {caad_eval_time:.2f} seconds")
        print(f"  - Samples evaluated: {len(caad_psnr)}")
        print(f"  - PSNR range: [{min(caad_psnr):.2f}, {max(caad_psnr):.2f}]")
        print(f"  - SSIM range: [{min(caad_ssim):.4f}, {max(caad_ssim):.4f}]")
        
        print(f"\nResults for {percent*100:.0f}% data:")
        print(f"  - Base Method: PSNR = {np.mean(base_psnr):.2f} ± {np.std(base_psnr):.2f}, SSIM = {np.mean(base_ssim):.4f} ± {np.std(base_ssim):.4f}")
        print(f"  - CAAD Method: PSNR = {np.mean(caad_psnr):.2f} ± {np.std(caad_psnr):.2f}, SSIM = {np.mean(caad_ssim):.4f} ± {np.std(caad_ssim):.4f}")
        print(f"  - Improvement: PSNR +{np.mean(caad_psnr) - np.mean(base_psnr):.2f}, SSIM +{np.mean(caad_ssim) - np.mean(base_ssim):.4f}")
        
        all_results.append({
            'percent': percent,
            'base_psnr': np.mean(base_psnr),
            'base_ssim': np.mean(base_ssim),
            'caad_psnr': np.mean(caad_psnr),
            'caad_ssim': np.mean(caad_ssim)
        })
    
    if len(percentages) > 1:
        print("\n" + "="*60)
        print("SUMMARY OF LIMITED DATA EXPERIMENTS")
        print("="*60)
        print(f"{'Data %':<10} {'Base PSNR':<12} {'Base SSIM':<12} {'CAAD PSNR':<12} {'CAAD SSIM':<12} {'PSNR Gain':<12} {'SSIM Gain':<12}")
        print("-" * 80)
        
        for result in all_results:
            p = result['percent']
            psnr_gain = result['caad_psnr'] - result['base_psnr']
            ssim_gain = result['caad_ssim'] - result['base_ssim']
            print(f"{p*100:<10.0f} {result['base_psnr']:<12.2f} {result['base_ssim']:<12.4f} "
                  f"{result['caad_psnr']:<12.2f} {result['caad_ssim']:<12.4f} "
                  f"{psnr_gain:<12.2f} {ssim_gain:<12.4f}")
        print("="*80)

def test_code():
    """
    Run a quick test of all experiments.
    This function executes a very brief version of each experiment.
    Execution should finish quickly.
    """
    print("\n" + "="*80)
    print("RUNNING QUICK TESTS OF ALL EXPERIMENTS")
    print("="*80)
    
    start_time = time.time()
    
    print("\nPreparing environment...")
    ensure_dirs_exist()
    print("  - Created necessary directories")
    
    print("\nSystem Information:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  - CPU count: {os.cpu_count()}")
    print(f"  - Random seed: {RANDOM_SEED}")
    
    print("\nTest Configuration:")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Test epochs: {TEST_EPOCHS}")
    print(f"  - Noise parameters: alpha={NOISE_ALPHA}, beta={NOISE_BETA}")
    
    print("\n" + "-"*80)
    print("[TEST 1/3] Experiment 1: Structured Noise and Denoising Quality")
    print("-"*80)
    exp1_start = time.time()
    experiment_structured_noise(test=True)
    exp1_time = time.time() - exp1_start
    print(f"Test 1 completed in {exp1_time:.2f} seconds")
    
    print("\n" + "-"*80)
    print("[TEST 2/3] Experiment 2: Reverse Diffusion Dynamics")
    print("-"*80)
    exp2_start = time.time()
    experiment_reverse_diffusion(test=True)
    exp2_time = time.time() - exp2_start
    print(f"Test 2 completed in {exp2_time:.2f} seconds")
    
    print("\n" + "-"*80)
    print("[TEST 3/3] Experiment 3: Limited Data Robustness")
    print("-"*80)
    exp3_start = time.time()
    experiment_limited_data(test=True)
    exp3_time = time.time() - exp3_start
    print(f"Test 3 completed in {exp3_time:.2f} seconds")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("ALL QUICK TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Test 1 (Structured Noise): {exp1_time:.2f} seconds ({exp1_time/total_time*100:.1f}%)")
    print(f"Test 2 (Reverse Diffusion): {exp2_time:.2f} seconds ({exp2_time/total_time*100:.1f}%)")
    print(f"Test 3 (Limited Data): {exp3_time:.2f} seconds ({exp3_time/total_time*100:.1f}%)")
    print("\nAll output files saved to logs/ directory")
    print("All model files saved to models/ directory")
    print("="*80)

def main():
    """Main entry point for the script."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Run CAAD experiments')
    parser.add_argument('--test', action='store_true', help='Run quick tests')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3], help='Run specific experiment (1, 2, or 3)')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COVARIANCE-ADJUSTED AMBIENT DIFFUSION (CAAD) EXPERIMENTS")
    print("="*80)
    
    ensure_dirs_exist()
    print("Created necessary directories: data/, logs/, models/, config/")
    
    print("\nSystem Information:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - CPU count: {os.cpu_count()}")
    print(f"  - Python version: {sys.version.split()[0]}")
    print(f"  - Operating system: {os.name} ({sys.platform})")
    
    print("\nExperiment Configuration:")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS} (full) / {TEST_EPOCHS} (test)")
    print(f"  - Noise parameters: alpha={NOISE_ALPHA}, beta={NOISE_BETA}")
    print(f"  - Diffusion parameters: iterations={DIFFUSION_ITERATIONS}, step_size={DIFFUSION_STEP_SIZE}")
    
    print("\nRunning selected experiment(s)...")
    
    if args.test:
        print("Mode: Quick test of all experiments")
        test_code()
    elif args.exp == 1:
        print("Mode: Full Experiment 1 - Structured Noise and Denoising Quality")
        experiment_structured_noise()
    elif args.exp == 2:
        print("Mode: Full Experiment 2 - Reverse Diffusion Dynamics")
        experiment_reverse_diffusion()
    elif args.exp == 3:
        print("Mode: Full Experiment 3 - Limited Data Robustness")
        experiment_limited_data()
    elif args.all:
        print("Mode: All Full Experiments")
        print("\nRunning Experiment 1...")
        experiment_structured_noise()
        print("\nRunning Experiment 2...")
        experiment_reverse_diffusion()
        print("\nRunning Experiment 3...")
        experiment_limited_data()
    else:
        print("Mode: Default (Quick test of all experiments)")
        test_code()
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print("All output files saved to logs/ directory")
    print("All model files saved to models/ directory")
    print("="*80)

if __name__ == '__main__':
    main()
