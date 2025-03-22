#!/usr/bin/env python3
"""
Main script for Score-Aligned Step Distillation (SASD) experiments.

This script orchestrates the entire workflow from data preprocessing to model training and evaluation.
It implements three experiments:
1. Ablation Study on the Dual-Loss Objective
2. Learnable Schedule vs. Fixed Schedule
3. Step Efficiency and Robustness Across Datasets

The script can be run in test mode for quick verification or in full mode for complete experiments.
"""

import torch
import os
import sys
import time
import argparse
import importlib.util
import numpy as np
from src.preprocess import prepare_data
from src.train import (
    experiment1 as train_experiment1,
    experiment2 as train_experiment2,
    experiment3 as train_experiment3,
    run_test_experiments as run_test_train
)
from src.evaluate import (
    evaluate_experiment1_results,
    evaluate_experiment2_results,
    evaluate_experiment3_results,
    generate_samples
)
from src.utils.visualization import plot_loss_curves, visualize_samples


def load_config(config_path):
    """
    Load configuration from a Python module.
    
    Args:
        config_path: Path to the configuration module
    
    Returns:
        dict: Configuration dictionary
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract all uppercase variables as configuration
    config = {name: getattr(config_module, name) for name in dir(config_module)
              if name.isupper() and not name.startswith('_')}
    
    return config


def setup_environment(config):
    """
    Set up the environment for experiments.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        device: PyTorch device
    """
    # Set random seed for reproducibility
    torch.manual_seed(config['RANDOM_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['RANDOM_SEED'])
    
    # Create necessary directories
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Set device
    device = torch.device(config['DEVICE'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def run_experiment1(config, device, test_mode=False):
    """
    Run Experiment 1: Ablation Study on the Dual-Loss Objective.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        test_mode: Whether to run in test mode
    
    Returns:
        dict: Experiment results
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: ABLATION STUDY ON THE DUAL-LOSS OBJECTIVE")
    print("="*80)
    
    # Use test configuration if in test mode
    if test_mode:
        exp_config = config['EXPERIMENT_CONFIG']['test_mode']
    else:
        exp_config = config['EXPERIMENT_CONFIG']['experiment1']
    
    # Run training
    start_time = time.time()
    results = train_experiment1(config, device)
    training_time = time.time() - start_time
    
    print(f"\nExperiment 1 training completed in {training_time:.2f} seconds")
    
    # Evaluate results
    eval_results = evaluate_experiment1_results(results, save_dir='./logs')
    
    # Plot loss curves
    loss_histories = {key: result['loss_history'] for key, result in results.items()}
    plot_loss_curves(
        loss_histories,
        title="Experiment 1: Ablation of Dual Loss",
        xlabel="Epoch",
        ylabel="Loss",
        save_path="./logs/experiment1_loss_curves.png"
    )
    
    # Print summary
    print("\nEXPERIMENT 1 SUMMARY:")
    print(f"Best λ value: {eval_results['best_lambda']}")
    print(f"Final losses: {eval_results['final_losses']}")
    
    return results


def run_experiment2(config, device, test_mode=False):
    """
    Run Experiment 2: Learnable Schedule vs. Fixed Schedule.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        test_mode: Whether to run in test mode
    
    Returns:
        dict: Experiment results
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: LEARNABLE SCHEDULE VS. FIXED SCHEDULE")
    print("="*80)
    
    # Use test configuration if in test mode
    if test_mode:
        exp_config = config['EXPERIMENT_CONFIG']['test_mode']
    else:
        exp_config = config['EXPERIMENT_CONFIG']['experiment2']
    
    # Run training
    start_time = time.time()
    results = train_experiment2(config, device)
    training_time = time.time() - start_time
    
    print(f"\nExperiment 2 training completed in {training_time:.2f} seconds")
    
    # Evaluate results
    eval_results = evaluate_experiment2_results(results, save_dir='./logs')
    
    # Plot loss curves
    loss_histories = {key: result['loss_history'] for key, result in results.items()}
    plot_loss_curves(
        loss_histories,
        title="Experiment 2: Fixed vs. Learnable Schedule",
        xlabel="Epoch",
        ylabel="Loss",
        save_path="./logs/experiment2_loss_curves.png"
    )
    
    # Print summary
    print("\nEXPERIMENT 2 SUMMARY:")
    print(f"Best configuration: {eval_results['best_config']}")
    print(f"Final losses: {eval_results['final_losses']}")
    
    return results


def run_experiment3(config, device, test_mode=False):
    """
    Run Experiment 3: Step Efficiency and Robustness Across Datasets.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        test_mode: Whether to run in test mode
    
    Returns:
        dict: Experiment results
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: STEP EFFICIENCY AND ROBUSTNESS ACROSS DATASETS")
    print("="*80)
    
    # Use test configuration if in test mode
    if test_mode:
        exp_config = config['EXPERIMENT_CONFIG']['test_mode']
    else:
        exp_config = config['EXPERIMENT_CONFIG']['experiment3']
    
    # Run training
    start_time = time.time()
    results = train_experiment3(config, device)
    training_time = time.time() - start_time
    
    print(f"\nExperiment 3 training completed in {training_time:.2f} seconds")
    
    # Evaluate results
    eval_results = evaluate_experiment3_results(results, save_dir='./logs')
    
    # Plot loss curves for each dataset and step configuration
    for dataset in eval_results['datasets']:
        dataset_results = {
            f"steps={steps}": results[f"{dataset}_steps{steps}"]['loss_history']
            for steps in eval_results['step_configs']
            if f"{dataset}_steps{steps}" in results
        }
        
        plot_loss_curves(
            dataset_results,
            title=f"Experiment 3: Step Efficiency on {dataset}",
            xlabel="Epoch",
            ylabel="Loss",
            save_path=f"./logs/experiment3_{dataset}_loss_curves.png"
        )
    
    # Print summary
    print("\nEXPERIMENT 3 SUMMARY:")
    print("Best Step Configuration per Dataset:")
    for i, dataset in enumerate(eval_results['datasets']):
        best_step_idx = np.argmin(eval_results['loss_matrix'][i, :])
        best_steps = eval_results['step_configs'][best_step_idx]
        best_loss = eval_results['loss_matrix'][i, best_step_idx]
        print(f"{dataset}: steps={best_steps} (Loss: {best_loss:.4f})")
    
    return results


def run_test_mode(config, device):
    """
    Run all experiments in test mode.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
    
    Returns:
        dict: Test results
    """
    print("\n" + "="*80)
    print("RUNNING ALL EXPERIMENTS IN TEST MODE")
    print("="*80)
    
    # Run test experiments
    start_time = time.time()
    results = run_test_train(config, device)
    test_time = time.time() - start_time
    
    print(f"\nAll test experiments completed in {test_time:.2f} seconds")
    
    return results


def main():
    """
    Main function to run SASD experiments.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run SASD experiments')
    parser.add_argument('--config', type=str, default='./config/sasd_config.py',
                        help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with minimal resources')
    parser.add_argument('--experiment', type=int, default=0,
                        help='Run specific experiment (1, 2, or 3), or 0 for all')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set up environment
    device = setup_environment(config)
    
    # Print experiment mode
    if args.test:
        print("Running in TEST mode (minimal resources)")
    else:
        print("Running in FULL mode")
    
    # Run experiments
    start_time = time.time()
    
    if args.experiment == 0:
        # Run all experiments
        if args.test:
            results = run_test_mode(config, device)
        else:
            results = {}
            results['experiment1'] = run_experiment1(config, device)
            results['experiment2'] = run_experiment2(config, device)
            results['experiment3'] = run_experiment3(config, device)
    elif args.experiment == 1:
        # Run only Experiment 1
        results = {'experiment1': run_experiment1(config, device, test_mode=args.test)}
    elif args.experiment == 2:
        # Run only Experiment 2
        results = {'experiment2': run_experiment2(config, device, test_mode=args.test)}
    elif args.experiment == 3:
        # Run only Experiment 3
        results = {'experiment3': run_experiment3(config, device, test_mode=args.test)}
    else:
        print(f"Invalid experiment number: {args.experiment}")
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    # Print overall summary
    print("\n" + "="*80)
    print("SASD EXPERIMENTS SUMMARY")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Configuration: {args.config}")
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"Device: {device}")
    
    print("\nExperiments completed successfully!")


if __name__ == '__main__':
    main()
