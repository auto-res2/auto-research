#!/usr/bin/env python3
"""
Main script for ACM optimizer experiments.

This script orchestrates the entire experiment process, using the preprocess.py,
train.py, and evaluate.py scripts to run the three experiments comparing the
performance of the ACM optimizer with SGD and Adam.
"""

import os
import sys
import yaml
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.preprocess import load_config, get_device
from src.evaluate import experiment1, experiment2, experiment3, run_all_experiments


def setup_logging(config):
    """
    Set up logging for the experiments.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        str: Path to the log file.
    """
    # Create log directory if it doesn't exist
    log_dir = config['general']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_log_{timestamp}.txt")
    
    # Redirect stdout and stderr to the log file and console
    class Logger:
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log = open(log_file, "w")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    
    return log_file


def print_system_info():
    """
    Print system information for reproducibility.
    """
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA: Not available")
    
    print(f"NumPy version: {np.__version__}")
    print("=========================\n")


def print_config_summary(config):
    """
    Print a summary of the configuration.
    
    Args:
        config (dict): Configuration dictionary.
    """
    print("=== Configuration Summary ===")
    print(f"Random seed: {config['general']['seed']}")
    print(f"Device: {config['general']['device']}")
    
    # Experiment 1
    print("\nExperiment 1:")
    print(f"  Dataset: {config['experiment1']['dataset']}")
    print(f"  Model: {config['experiment1']['model']}")
    print(f"  Batch size: {config['experiment1']['batch_size']}")
    print(f"  Epochs: {config['experiment1']['num_epochs']}")
    
    # Experiment 2
    print("\nExperiment 2:")
    print(f"  Synthetic loss: f(x, y) = {config['experiment2']['synthetic_loss']['a']}*x^2 + {config['experiment2']['synthetic_loss']['b']}*y^2")
    print(f"  Steps: {config['experiment2']['num_steps']}")
    
    # Experiment 3
    print("\nExperiment 3:")
    print(f"  Dataset: {config['experiment3']['dataset']}")
    print(f"  Model: {config['experiment3']['model']}")
    print(f"  Batch size: {config['experiment3']['batch_size']}")
    print(f"  Epochs: {config['experiment3']['num_epochs']}")
    print(f"  Trials: {config['experiment3']['n_trials']}")
    
    # Test run settings
    if config['test_run']['enabled']:
        print("\nTest run enabled with reduced settings:")
        print(f"  Experiment 1 epochs: {config['test_run']['experiment1']['num_epochs']}")
        print(f"  Experiment 2 steps: {config['test_run']['experiment2']['num_steps']}")
        print(f"  Experiment 3 epochs: {config['test_run']['experiment3']['num_epochs']}")
        print(f"  Experiment 3 trials: {config['test_run']['experiment3']['n_trials']}")
    
    print("============================\n")


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run ACM optimizer experiments')
    parser.add_argument('--config', type=str, default='config/acm_experiments.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--test', action='store_true',
                        help='Run a test with reduced dataset size, epochs, and trials')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3],
                        help='Run only the specified experiment (1, 2, or 3)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    """
    Main function to run the experiments.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        config['general']['seed'] = args.seed
    
    # Set up logging
    log_file = setup_logging(config)
    
    # Print header
    print("=" * 80)
    print("ACM Optimizer Experiments")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print()
    
    # Print system information
    print_system_info()
    
    # Print configuration summary
    print_config_summary(config)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['general']['seed'])
    
    # Create directories if they don't exist
    os.makedirs(config['general']['log_dir'], exist_ok=True)
    os.makedirs(config['general']['model_dir'], exist_ok=True)
    os.makedirs(config['general']['data_dir'], exist_ok=True)
    
    # Get device
    device = get_device(config)
    
    # Run experiments
    start_time = time.time()
    
    if args.experiment is None:
        # Run all experiments
        print("Running all experiments...")
        exp1_results, exp2_results, exp3_results = run_all_experiments(
            args.config, args.test
        )
    else:
        # Run only the specified experiment
        if args.experiment == 1:
            print("Running Experiment 1 only...")
            exp1_results = experiment1(config, args.test)
        elif args.experiment == 2:
            print("Running Experiment 2 only...")
            exp2_results = experiment2(config, args.test)
        elif args.experiment == 3:
            print("Running Experiment 3 only...")
            exp3_results = experiment3(config, args.test)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Print completion message
    print("\n" + "=" * 80)
    print("Experiments completed successfully!")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results and logs saved to: {config['general']['log_dir']}")
    if config['experiment1'].get('save_model', False) or config['experiment3'].get('save_model', False):
        print(f"Models saved to: {config['general']['model_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
