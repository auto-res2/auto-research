"""
Main script for running MML-BO experiments.
"""
import numpy as np
import torch
import os
import time
import sys
from importlib import import_module

from preprocess import create_synthetic_functions, create_meta_learning_data
from train import optimize_baseline, optimize_mml_bo, train_task_encoder
from evaluate import experiment1, experiment2, experiment3

def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, running on CPU")

def run_experiments(test_mode=False):
    """
    Run all experiments for MML-BO research.
    
    Args:
        test_mode (bool): If True, run in test mode with reduced iterations
    """
    start_time = time.time()
    
    print("\n===== Creating Directories =====")
    os.makedirs('logs', exist_ok=True)
    print("✓ Created logs directory")
    os.makedirs('models', exist_ok=True)
    print("✓ Created models directory")
    os.makedirs('data', exist_ok=True)
    print("✓ Created data directory")
    
    print("\n===== System Information =====")
    print_gpu_info()
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    print("\n===== Loading Configuration =====")
    try:
        config_module = import_module('config.mml_bo_config')
        exp1_config = config_module.EXP1_CONFIG
        exp2_config = config_module.EXP2_CONFIG
        exp3_config = config_module.EXP3_CONFIG
        device_config = config_module.DEVICE_CONFIG
        print("✓ Successfully loaded configuration from config/mml_bo_config.py")
        print(f"✓ Device config: {'GPU' if device_config['use_gpu'] else 'CPU'}, Memory limit: {device_config['gpu_memory_limit']}GB")
    except ImportError:
        print("⚠ Warning: Config module not found. Using default configurations.")
        exp1_config = {'iters': 100, 'init_val': [0.0], 'levels': 3, 'step_size': 0.1, 'noise_std_quad': 0.1, 'noise_std_sin': 1.0}
        exp2_config = {'iters': 50, 'levels': 3, 'fixed_tau': 0.1}
        exp3_config = {'epochs': 100, 'lr': 0.01, 'batch_size': 64, 'embedding_dim': 1, 'n_samples': 100, 'feature_dim': 10}
        device_config = {'use_gpu': True, 'gpu_memory_limit': 16}
    
    if test_mode:
        print("\n===== Running in TEST MODE with reduced iterations =====")
        print(f"⚠ Reducing iterations from {exp1_config['iters']} to 10 for Experiment 1")
        exp1_config['iters'] = 10
        print(f"⚠ Reducing iterations from {exp2_config['iters']} to 5 for Experiment 2")
        exp2_config['iters'] = 5
        print(f"⚠ Reducing epochs from {exp3_config['epochs']} to 5 for Experiment 3")
        exp3_config['epochs'] = 5
    
    print("\n===== Starting MML-BO Experiments =====")
    print("Experiment 1 Configuration:")
    for key, value in exp1_config.items():
        print(f"  - {key}: {value}")
    print("\nExperiment 2 Configuration:")
    for key, value in exp2_config.items():
        print(f"  - {key}: {value}")
    print("\nExperiment 3 Configuration:")
    for key, value in exp3_config.items():
        print(f"  - {key}: {value}")
    
    print("\n===== Preparing Synthetic Functions =====")
    funcs = create_synthetic_functions()
    print("✓ Created quadratic function with noise std:", exp1_config['noise_std_quad'])
    print("✓ Created sinusoidal function with noise std:", exp1_config['noise_std_sin'])
    
    print("\n===== Generating Meta-Learning Data =====")
    data, target = create_meta_learning_data(
        n_samples=exp3_config['n_samples'],
        feature_dim=exp3_config['feature_dim']
    )
    print(f"✓ Generated {exp3_config['n_samples']} samples with {exp3_config['feature_dim']} features")
    print(f"✓ Data shape: {data.shape}, Target shape: {target.shape}")
    
    print("\n===== Running Experiment 1: Synthetic Multi-Task Optimization =====")
    print(f"Running with {exp1_config['iters']} iterations")
    print(f"Initial value: {exp1_config['init_val']}")
    print(f"Multi-level steps: {exp1_config['levels']}")
    print(f"Step size: {exp1_config['step_size']}")
    exp1_results = experiment1(funcs, exp1_config)
    
    print("\n===== Experiment 1 Results =====")
    print(f"Best value found (Quadratic) - Baseline: {min(exp1_results['quadratic']['baseline']):.4f}")
    print(f"Best value found (Quadratic) - MML-BO: {min(exp1_results['quadratic']['mml_bo']):.4f}")
    print(f"Improvement: {(min(exp1_results['quadratic']['baseline']) - min(exp1_results['quadratic']['mml_bo'])):.4f}")
    print(f"Best value found (Sinusoidal) - Baseline: {min(exp1_results['sinusoidal']['baseline']):.4f}")
    print(f"Best value found (Sinusoidal) - MML-BO: {min(exp1_results['sinusoidal']['mml_bo']):.4f}")
    print(f"Improvement: {(min(exp1_results['sinusoidal']['baseline']) - min(exp1_results['sinusoidal']['mml_bo'])):.4f}")
    
    print("\n===== Running Experiment 2: Adaptive Exploration-Exploitation =====")
    print(f"Running with {exp2_config['iters']} iterations")
    print(f"Fixed tau value: {exp2_config['fixed_tau']}")
    print(f"Multi-level steps: {exp2_config['levels']}")
    exp2_results = experiment2(exp2_config)
    
    print("\n===== Experiment 2 Results =====")
    print(f"Average fixed exploration parameter: {sum(exp2_results['fixed'])/len(exp2_results['fixed']):.4f}")
    print(f"Average adaptive exploration parameter: {sum(exp2_results['adaptive'])/len(exp2_results['adaptive']):.4f}")
    print(f"Min adaptive tau: {min(exp2_results['tau_alloc']):.4f}")
    print(f"Max adaptive tau: {max(exp2_results['tau_alloc']):.4f}")
    
    print("\n===== Running Experiment 3: Meta-Learned Task Embeddings =====")
    print(f"Running with {exp3_config['epochs']} epochs")
    print(f"Learning rate: {exp3_config['lr']}")
    print(f"Batch size: {exp3_config['batch_size']}")
    print(f"Embedding dimension: {exp3_config['embedding_dim']}")
    exp3_results = experiment3(data, target, exp3_config)
    
    print("\n===== Experiment 3 Results =====")
    print(f"Final loss (Unimodal): {exp3_results['unimodal_loss'][-1]:.4f}")
    print(f"Final loss (Richer uncertainty): {exp3_results['richer_loss'][-1]:.4f}")
    print(f"Loss improvement: {(exp3_results['unimodal_loss'][-1] - exp3_results['richer_loss'][-1]):.4f}")
    
    total_time = time.time() - start_time
    print("\n===== Experiment Summary =====")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("All experiments completed successfully")
    print("PDF figures saved to logs/ directory:")
    print("  - logs/training_loss_quadratic.pdf")
    print("  - logs/training_loss_sinusoidal.pdf")
    print("  - logs/accuracy_adaptive_vs_fixed.pdf")
    print("  - logs/accuracy_mlmc_uncertainty_alloc_pair1.pdf")
    print("  - logs/training_loss_meta_uncertainty.pdf")
    
    return {
        'exp1': exp1_results,
        'exp2': exp2_results,
        'exp3': exp3_results
    }

if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    
    results = run_experiments(test_mode=test_mode)
    
    print("\n===== MML-BO Research Completed =====")
