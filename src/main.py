"""
Main script for running MML-BO experiments.
"""
import numpy as np
import torch
import os
import time
import sys
from importlib import import_module

from src.preprocess import create_synthetic_functions, create_meta_learning_data
from src.train import optimize_baseline, optimize_mml_bo, train_task_encoder
from src.evaluate import experiment1, experiment2, experiment3

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
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("\n===== System Information =====")
    print_gpu_info()
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        config_module = import_module('config.mml_bo_config')
        exp1_config = config_module.EXP1_CONFIG
        exp2_config = config_module.EXP2_CONFIG
        exp3_config = config_module.EXP3_CONFIG
        device_config = config_module.DEVICE_CONFIG
    except ImportError:
        print("Warning: Config module not found. Using default configurations.")
        exp1_config = {'iters': 100, 'init_val': [0.0], 'levels': 3, 'step_size': 0.1, 'noise_std_quad': 0.1, 'noise_std_sin': 1.0}
        exp2_config = {'iters': 50, 'levels': 3, 'fixed_tau': 0.1}
        exp3_config = {'epochs': 100, 'lr': 0.01, 'batch_size': 64, 'embedding_dim': 1, 'n_samples': 100, 'feature_dim': 10}
        device_config = {'use_gpu': True, 'gpu_memory_limit': 16}
    
    if test_mode:
        print("\n===== Running in TEST MODE with reduced iterations =====")
        exp1_config['iters'] = 10
        exp2_config['iters'] = 5
        exp3_config['epochs'] = 5
    
    print("\n===== Starting MML-BO Experiments =====")
    print(f"Configuration: {exp1_config}, {exp2_config}, {exp3_config}")
    
    funcs = create_synthetic_functions()
    
    data, target = create_meta_learning_data(
        n_samples=exp3_config['n_samples'],
        feature_dim=exp3_config['feature_dim']
    )
    
    print("\n")
    exp1_results = experiment1(funcs, exp1_config)
    
    print("\n")
    exp2_results = experiment2(exp2_config)
    
    print("\n")
    exp3_results = experiment3(data, target, exp3_config)
    
    total_time = time.time() - start_time
    print("\n===== Experiment Summary =====")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("All experiments completed successfully")
    print("PDF figures saved to logs/ directory")
    
    return {
        'exp1': exp1_results,
        'exp2': exp2_results,
        'exp3': exp3_results
    }

if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    
    results = run_experiments(test_mode=test_mode)
    
    print("\n===== MML-BO Research Completed =====")
