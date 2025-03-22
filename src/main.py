"""
Main script for running TCPGS experiments.
"""

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Import configuration
import sys
import os

# Add the repository root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.tcpgs_config import (
    DEVICE, DATASET, BATCH_SIZE, 
    DENOISE_STEPS, NOISE_LEVELS, EXPERIMENTS
)

# Import modules
from preprocess import get_dataset, get_initial_noise
from train import BaseDiffusionModel, TCPGSDiffusionModel
from evaluate import (
    experiment_robustness, 
    experiment_convergence, 
    experiment_ablation
)


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_environment():
    """Setup the environment for experiments."""
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Print GPU memory information
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    return device


def instantiate_models():
    """Create model instances for experiments."""
    # Initialize the models
    base_model = BaseDiffusionModel()
    tcpgs_model = TCPGSDiffusionModel(use_consistency=True)
    tcpgs_no_consistency_model = TCPGSDiffusionModel(use_consistency=False)
    
    # Group models for different experiments
    models = {
        'BaseMethod': base_model,
        'TCPGS': tcpgs_model
    }
    
    model_variants = {
        'BaseMethod': base_model,
        'TCPGS_no_consistency': tcpgs_no_consistency_model,
        'Full_TCPGS': tcpgs_model
    }
    
    return models, model_variants


def run_experiments(models, model_variants, train_loader, device):
    """Run all three experiments."""
    results = {}
    
    # Experiment 1: Robustness to corrupted data and variable noise levels
    print(f"\n{'='*80}\nRunning Experiment: {EXPERIMENTS['robustness']}\n{'='*80}")
    results['robustness'] = experiment_robustness(
        models, train_loader, device=device, noise_levels=NOISE_LEVELS
    )
    
    # Experiment 2: Convergence efficiency
    print(f"\n{'='*80}\nRunning Experiment: {EXPERIMENTS['convergence']}\n{'='*80}")
    results['convergence'] = experiment_convergence(
        models, device=device, step_counts=list(DENOISE_STEPS.values())
    )
    
    # Experiment 3: Ablation study on Tweedie consistency
    print(f"\n{'='*80}\nRunning Experiment: {EXPERIMENTS['ablation']}\n{'='*80}")
    results['ablation'] = experiment_ablation(
        model_variants, train_loader, device=device
    )
    
    return results


def display_summary(results):
    """Display summary of experimental results."""
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    # Robustness results
    print("\n1. Robustness to Corrupted Data:")
    print("-" * 60)
    print(f"{'Model':<20} {'Noise Type':<15} {'Level':<8} {'FID Score':<10}")
    print("-" * 60)
    
    for (model, noise_key), fid in results['robustness'].items():
        noise_parts = noise_key.split('_')
        noise_type = noise_parts[0]
        noise_level = noise_parts[1]
        print(f"{model:<20} {noise_type:<15} {noise_level:<8} {fid:<10.2f}")
    
    # Convergence results
    print("\n2. Convergence Efficiency:")
    print("-" * 60)
    print(f"{'Model':<20} {'Steps':<8} {'FID Score':<10} {'Avg Time (s)':<15}")
    print("-" * 60)
    
    for (model, steps), result in results['convergence'].items():
        print(f"{model:<20} {steps:<8} {result['FID']:<10.2f} {result['AvgTime']:<15.4f}")
    
    # Ablation results
    print("\n3. Tweedie Consistency Ablation:")
    print("-" * 60)
    print(f"{'Model Variant':<25} {'MSE Loss':<10}")
    print("-" * 60)
    
    for variant, mse in results['ablation'].items():
        print(f"{variant:<25} {mse:<10.4f}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)


def main():
    """Main function to run the TCPGS experiments."""
    print("\n" + "="*80)
    print("TCPGS EXPERIMENT SUITE")
    print("="*80)
    print(f"Starting experiment at: {os.path.basename(__file__)}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*80)
    
    # Set random seeds for reproducibility
    print("\n[1/7] Setting random seeds for reproducibility...")
    set_seeds()
    print("✓ Seeds set successfully")
    
    # Setup environment
    print("\n[2/7] Setting up environment...")
    device = setup_environment()
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("⚠ WARNING: CUDA requested but not available. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)
    print(f"✓ Environment setup complete. Using device: {device}")
    
    print("\n[3/7] Loading dataset: {DATASET}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Dataset: {DATASET}")
    train_loader = get_dataset(dataset_name=DATASET, batch_size=BATCH_SIZE)
    print(f"✓ Dataset loaded successfully with {len(train_loader)} batches")
    
    print("\n[4/7] Instantiating models...")
    models, model_variants = instantiate_models()
    print("✓ Models created:")
    for name in models.keys():
        print(f"  - {name}")
    print("✓ Model variants created:")
    for name in model_variants.keys():
        print(f"  - {name}")
    
    print("\n[5/7] Running experiments...")
    print(f"- Experiments to run: {', '.join(EXPERIMENTS.keys())}")
    results = run_experiments(models, model_variants, train_loader, device)
    print("✓ All experiments completed successfully")
    
    print("\n[6/7] Displaying results summary...")
    display_summary(results)
    
    print("\n[7/7] Saving models...")
    for name, model in models.items():
        model_path = f"models/{name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved: {model_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()
