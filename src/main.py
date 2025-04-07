
import torch
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.makedirs('logs', exist_ok=True)

from preprocess import load_cifar10
from train import DiffusionModel, DummyClassifier
from evaluate import experiment_ablation, experiment_robustness, experiment_adaptive_control

sys.path.append('config')
try:
    from cedp_config import EXPERIMENT_CONFIG, CEDP_CONFIG, GPU_CONFIG
except ImportError:
    print("Configuration not found. Using default values.")
    EXPERIMENT_CONFIG = {'batch_size': 32, 'seed': 42, 'max_samples': 128}
    CEDP_CONFIG = {'noise_level': 0.3, 'iterations': 5, 'consistency_threshold': 0.01, 'adaptive_decay_factor': 0.9}
    GPU_CONFIG = {'precision': 'float32'}

def setup_environment():
    """Configure the environment for the experiments."""
    seed = EXPERIMENT_CONFIG.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA not available. Using CPU.")
    
    return device

def try_import_optional_modules():
    """Try to import optional modules and return availability flags."""
    has_torchattacks = False
    
    try:
        import torchattacks
        has_torchattacks = True
        print("torchattacks is available for adversarial attack generation.")
    except ImportError:
        print("torchattacks not found; will simulate adversarial perturbations.")
    
    return {
        'torchattacks': has_torchattacks
    }

def run_experiments():
    """Run the CEDP experiments."""
    start_time = time.time()
    
    device = setup_environment()
    
    available_modules = try_import_optional_modules()
    
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(batch_size=EXPERIMENT_CONFIG.get('batch_size', 32))
    print(f"Dataset loaded. Test set size: {len(test_loader.dataset)} samples.")
    
    print("Initializing models...")
    diffusion_model = DiffusionModel().to(device)
    diffusion_model.eval()
    classifier = DummyClassifier().to(device)
    classifier.eval()
    
    print("\nExperiment Configuration:")
    print(f"Batch Size: {EXPERIMENT_CONFIG.get('batch_size', 32)}")
    print(f"Seed: {EXPERIMENT_CONFIG.get('seed', 42)}")
    print(f"Max Samples: {EXPERIMENT_CONFIG.get('max_samples', 128)}")
    print(f"Noise Level: {CEDP_CONFIG.get('noise_level', 0.3)}")
    print(f"Iterations: {CEDP_CONFIG.get('iterations', 5)}")
    print(f"Consistency Threshold: {CEDP_CONFIG.get('consistency_threshold', 0.01)}")
    print(f"GPU Precision: {GPU_CONFIG.get('precision', 'float32')}")
    
    print("\nStarting experiments...")
    
    ablation_results = experiment_ablation(device, diffusion_model, test_loader)
    
    robustness_results = experiment_robustness(device, diffusion_model, classifier, test_loader)
    
    adaptive_results = experiment_adaptive_control(device, diffusion_model, test_loader)
    
    print("\n=== Summary of Results ===")
    print("Experiment 1 (Ablation Study):")
    print(f"  PSNR improvement (CEDP vs Base): {ablation_results['psnr']['cedp'] - ablation_results['psnr']['base']:.2f} dB")
    print(f"  SSIM improvement (CEDP vs Base): {ablation_results['ssim']['cedp'] - ablation_results['ssim']['base']:.4f}")
    print(f"  PSNR values: Base={ablation_results['psnr']['base']:.2f}, Dual={ablation_results['psnr']['dual']:.2f}, CEDP={ablation_results['psnr']['cedp']:.2f}")
    print(f"  SSIM values: Base={ablation_results['ssim']['base']:.4f}, Dual={ablation_results['ssim']['dual']:.4f}, CEDP={ablation_results['ssim']['cedp']:.4f}")
    
    print("\nExperiment 2 (Adversarial Robustness):")
    print(f"  Accuracy improvement (CEDP vs Base): {robustness_results['accuracy']['cedp'] - robustness_results['accuracy']['base']:.2f}%")
    print(f"  Accuracy values: Base={robustness_results['accuracy']['base']:.2f}%, CEDP={robustness_results['accuracy']['cedp']:.2f}%")
    
    print("\nExperiment 3 (Adaptive Randomness Control):")
    print(f"  PSNR improvement (Adaptive vs Fixed): {adaptive_results['psnr']['adaptive'] - adaptive_results['psnr']['fixed']:.2f} dB")
    print(f"  Time comparison: Adaptive={adaptive_results['time']['adaptive']:.4f}s, Fixed={adaptive_results['time']['fixed']:.4f}s")
    print(f"  Noise level adaptation: {adaptive_results['noise_records']}")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    return {
        "ablation": ablation_results,
        "robustness": robustness_results,
        "adaptive": adaptive_results,
        "execution_time": total_time
    }

if __name__ == "__main__":
    print("=" * 80)
    print("Consistency-Enhanced Diffusion Purification (CEDP) Experiments")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    results = run_experiments()
    
    print("-" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
