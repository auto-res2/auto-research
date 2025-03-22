"""
Main script for running TCPGS experiments.
"""

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Import configuration
from config.tcpgs_config import (
    DEVICE, DATASET, BATCH_SIZE, 
    DENOISE_STEPS, NOISE_LEVELS, EXPERIMENTS
)

# Import modules
from src.preprocess import get_dataset, get_initial_noise
from src.train import BaseDiffusionModel, TCPGSDiffusionModel
from src.evaluate import (
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
    
    # Set random seeds for reproducibility
    set_seeds()
    
    # Setup environment
    device = setup_environment()
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)
    
    print(f"\nLoading dataset: {DATASET}")
    train_loader = get_dataset(dataset_name=DATASET, batch_size=BATCH_SIZE)
    
    print("\nInstantiating models...")
    models, model_variants = instantiate_models()
    
    print("\nRunning experiments...")
    results = run_experiments(models, model_variants, train_loader, device)
    
    print("\nDisplaying results summary...")
    display_summary(results)
    
    print("\nSaving models...")
    for name, model in models.items():
        torch.save(model.state_dict(), f"models/{name}.pt")
        print(f"Model saved: models/{name}.pt")
    
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
