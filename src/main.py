"""
Main script for running ABS-Diff experiments.

This script orchestrates the execution of the three experiments for the
Adaptive Bayesian SDE-Guided Diffusion (ABS-Diff) method.
"""

import os
import torch
import yaml
import pytorch_lightning as pl
from tqdm import tqdm

# Import from other files
from src.preprocess import load_cifar10, set_random_seed
from src.train import (
    ABS_DiffModule, DynamicDiffusionModel, RegimeClassifier, 
    dynamic_noise_update, fixed_noise_schedule, train_model
)
from src.evaluate import test_dynamic_noise_conditioning, compare_sde_regimes


def print_header(header):
    """
    Print a formatted header for experiments.
    
    Args:
        header: Header text
    """
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))


def load_config(config_path="config/abs_diff_config.yaml"):
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        config: Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment1(train_loader, num_epochs=1):
    """
    Run Experiment 1: Regime Adaptive Classifier Ablation Study.
    
    Args:
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
    """
    print_header("Experiment 1: Regime Adaptive Classifier Ablation Study")
    
    # Adaptive variant run
    print("\nRunning adaptive variant (full ABS-Diff)...")
    adaptive_module = train_model(train_loader, num_epochs, adaptive=True)
    print("Adaptive variant training complete.")
    
    # Fixed variant run
    print("\nRunning fixed solver variant (ablated: no regime classifier)...")
    fixed_module = train_model(train_loader, num_epochs, adaptive=False)
    print("Fixed solver variant training complete.")


def run_experiment2():
    """
    Run Experiment 2: Dynamic Noise Conditioning vs. Fixed Noise Scheduling.
    """
    print_header("Experiment 2: Dynamic Noise Conditioning vs. Fixed Noise Scheduling")
    
    print("Preparing for noise conditioning experiment...")
    regime_classifier = RegimeClassifier()
    adaptive_model = DynamicDiffusionModel(regime_classifier)
    fixed_model = DynamicDiffusionModel(regime_classifier)
    
    # Test dynamic vs fixed noise conditioning
    test_dynamic_noise_conditioning(regime_classifier, adaptive_model, fixed_model)
    

def run_experiment3():
    """
    Run Experiment 3: Comparative Study of Bayesian SDE Solvers Across Regimes.
    """
    print_header("Experiment 3: Bayesian SDE Solvers Across Regimes")
    
    # Compare SDE regimes
    compare_sde_regimes()


def test_experiments():
    """
    Run a quick test of all experiments to verify proper execution.
    """
    print_header("TEST: Running quick experiments to verify code execution")
    
    # Load data
    print("Loading CIFAR-10 training data ...")
    train_loader, _ = load_cifar10(batch_size=64, num_workers=2)
    
    # Experiment 1: Run one epoch
    print("\n[Experiment 1 Test] Running training for 1 epoch (batch logging enabled)...")
    run_experiment1(train_loader, num_epochs=1)
    
    # Experiment 2: Execute noise scheduling comparison
    print("\n[Experiment 2 Test] Running dynamic vs fixed noise schedule check ...")
    run_experiment2()
    
    # Experiment 3: Run reverse SDE simulation
    print("\n[Experiment 3 Test] Running reverse SDE simulation (10 steps) ...")
    run_experiment3()
    
    print_header("All tests completed successfully.")


def main():
    """
    Main function to run the ABS-Diff experiments.
    """
    # Load configuration
    config = load_config()
    
    # Set random seed for reproducibility
    set_random_seed(config['training']['seed'])
    
    # Create directories if needed
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data
    train_loader, test_loader = load_cifar10(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Determine whether to run in test mode
    test_mode = config['experiments']['test_mode']
    
    if test_mode:
        # Run quick tests of all experiments
        test_experiments()
    else:
        # Run full experiments
        if config['experiments']['run_experiment1']:
            run_experiment1(train_loader, num_epochs=config['training']['num_epochs'])
            
        if config['experiments']['run_experiment2']:
            run_experiment2()
            
        if config['experiments']['run_experiment3']:
            run_experiment3()


if __name__ == '__main__':
    main()
