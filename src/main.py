#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements the main entry point for the BI-SDICL (Bias-Integrated Sequential 
Decision In-Context Learner) experiments. It orchestrates the entire process from model 
training to evaluation using the components defined in preprocess.py, train.py, and evaluate.py.

The script runs three experiments:
1. Robustness under Environmental Stochasticity
2. Ablation Study of the Bias Conversion Module
3. Interpretability and Diagnostic Visualization of Bias Integration
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from datetime import datetime

# Import components from other modules
from preprocess import prepare_environment, generate_demonstration_tokens
from train import create_policy, save_model
from evaluate import (
    experiment_robustness, 
    experiment_ablation, 
    experiment_interpretability,
    run_all_experiments
)

def set_random_seed(seed):
    """
    Sets random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set deterministic behavior for CuDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directories():
    """
    Creates necessary directories if they don't exist.
    """
    directories = ['logs', 'models', 'config']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def print_gpu_info():
    """
    Prints information about available GPUs.
    """
    if torch.cuda.is_available():
        print("\n=== GPU Information ===")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("\n=== No GPU Available ===")
        print("Running on CPU only")

def print_experiment_header():
    """
    Prints a header for the experiment.
    """
    print("\n" + "="*80)
    print("BI-SDICL: Bias-Integrated Sequential Decision In-Context Learner Experiments")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print_gpu_info()
    print("="*80 + "\n")

def parse_arguments():
    """
    Parses command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='BI-SDICL Experiments')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with fewer episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--experiment', type=int, default=0, 
                        help='Experiment to run (0: all, 1: robustness, 2: ablation, 3: interpretability)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run experiments on (cuda or cpu)')
    return parser.parse_args()

def run_experiment(args):
    """
    Runs the specified experiment(s).
    
    Args:
        args: Command line arguments
    """
    print_experiment_header()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Create necessary directories
    create_directories()
    
    # Set device for PyTorch
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Run experiments
    if args.experiment == 0:
        print("\nRunning all experiments...")
        results = run_all_experiments(quick_test=args.quick_test)
        
        # Print summary of results
        print("\n=== Summary of Results ===")
        print("Experiment 1 (Robustness):")
        for noise, data in results['experiment1_robustness'].items():
            print(f"  Noise level {noise}: Avg reward = {data['avg_eval_reward']:.2f}")
        
        print("\nExperiment 2 (Ablation):")
        print(f"  Full BI-SDICL: Avg reward = {results['experiment2_ablation']['full_bi_sdicl']['avg_eval_reward']:.2f}")
        print(f"  No Conversion: Avg reward = {results['experiment2_ablation']['no_conversion']['avg_eval_reward']:.2f}")
        print(f"  Base Method: Avg reward = {results['experiment2_ablation']['base_method']['avg_eval_reward']:.2f}")
        
        print("\nExperiment 3 (Interpretability):")
        print(f"  Original Bias: Avg reward = {results['experiment3_interpretability']['avg_baseline_reward']:.2f}")
        print(f"  Perturbed Bias: Avg reward = {results['experiment3_interpretability']['avg_perturbed_reward']:.2f}")
        
    elif args.experiment == 1:
        print("\nRunning Experiment 1: Robustness under Environmental Stochasticity...")
        if args.quick_test:
            noise_levels = [0.0, 0.2]  # Fewer noise levels for quick test
        else:
            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        results = experiment_robustness(noise_levels=noise_levels, 
                                        num_episodes=5 if args.quick_test else 20, 
                                        seed=args.seed)
        
    elif args.experiment == 2:
        print("\nRunning Experiment 2: Ablation Study of the Bias Conversion Module...")
        results = experiment_ablation(noise_level=0.2, 
                                     num_episodes=5 if args.quick_test else 20, 
                                     seed=args.seed)
        
    elif args.experiment == 3:
        print("\nRunning Experiment 3: Interpretability and Diagnostic Visualization...")
        results = experiment_interpretability(noise_level=0.2, 
                                             num_episodes=5 if args.quick_test else 20, 
                                             seed=args.seed)
    
    print("\n=== Experiment Completed ===")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Results saved to logs/ directory")
    print("="*80 + "\n")

def test_code():
    """
    A quick test function to verify that the implementation runs.
    Each experiment is run with very few episodes so that the test finishes quickly.
    """
    print("Running quick tests for BI‑SDICL experiments.")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create necessary directories
    create_directories()
    
    # For quick tests, we reduce the number of episodes.
    # Experiment 1: Robustness test at two noise levels.
    print("\nRunning Experiment 1: Robustness Under Environmental Stochasticity (Quick Test)")
    noise_levels = [0.0, 0.2]
    results_exp1 = experiment_robustness(noise_levels=noise_levels, num_episodes=2, seed=42)
    
    # Experiment 2: Quick ablation test.
    print("\nRunning Experiment 2: Ablation Study (Quick Test)")
    results_exp2 = experiment_ablation(noise_level=0.2, num_episodes=2, seed=42)
    
    # Experiment 3: Quick interpretability test.
    print("\nRunning Experiment 3: Interpretability and Bias Diagnostic Visualization (Quick Test)")
    results_exp3 = experiment_interpretability(noise_level=0.2, num_episodes=2, seed=42)
    
    print("\nQuick tests finished successfully.")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    if args.quick_test:
        # Run quick test
        test_code()
    else:
        # Run full experiment
        run_experiment(args)
