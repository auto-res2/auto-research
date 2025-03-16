#!/usr/bin/env python3
"""
Test run script for ACM optimizer experiments.

This script runs a simplified version of the experiments with reduced epochs
and dataset sizes to quickly verify that the implementation works correctly.
"""

import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime

from preprocess import load_config, get_device
from evaluate import experiment1, experiment2, experiment3


def setup_test_config(config_path='config/acm_experiments.yaml'):
    """
    Set up a test configuration with reduced settings.
    
    Args:
        config_path (str): Path to the original configuration file.
        
    Returns:
        dict: Test configuration dictionary.
    """
    # Load the original configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure test_run is enabled
    config['test_run']['enabled'] = True
    
    # Reduce settings for faster testing
    config['test_run']['experiment1']['num_epochs'] = 2
    config['test_run']['experiment1']['batch_size'] = 64
    
    config['test_run']['experiment2']['num_steps'] = 20
    
    config['test_run']['experiment3']['num_epochs'] = 1
    config['test_run']['experiment3']['n_trials'] = 2
    config['test_run']['experiment3']['batch_size'] = 64
    
    return config


def test_experiment1(config):
    """
    Test Experiment 1: Convergence Speed and Generalization on CIFAR-10.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        bool: True if the test passed, False otherwise.
    """
    print("\n=== Testing Experiment 1 ===")
    try:
        # Add debug prints to track the issue
        print("Debug: Starting experiment1 with test_run=True")
        
        # Run experiment1 with test_run=True
        results = experiment1(config, test_run=True)
        
        print("Debug: experiment1 returned results")
        print(f"Debug: Results type: {type(results)}")
        
        # Unpack results
        losses_acm, losses_sgd, losses_adam, acc_acm, acc_sgd, acc_adam = results
        
        print(f"Debug: Unpacked results")
        print(f"Debug: losses_acm type: {type(losses_acm)}")
        print(f"Debug: losses_sgd type: {type(losses_sgd)}")
        print(f"Debug: losses_adam type: {type(losses_adam)}")
        print(f"Debug: acc_acm type: {type(acc_acm)}")
        print(f"Debug: acc_sgd type: {type(acc_sgd)}")
        print(f"Debug: acc_adam type: {type(acc_adam)}")
        
        # Convert all results to lists to ensure compatibility
        losses_acm = list(map(float, losses_acm))
        losses_sgd = list(map(float, losses_sgd))
        losses_adam = list(map(float, losses_adam))
        acc_acm = list(map(float, acc_acm))
        acc_sgd = list(map(float, acc_sgd))
        acc_adam = list(map(float, acc_adam))
        
        print("Debug: Converted all results to float lists")
        
        # Basic validation
        print(f"Debug: Expected epochs: {config['test_run']['experiment1']['num_epochs']}")
        print(f"Debug: Actual losses_acm length: {len(losses_acm)}")
        
        assert len(losses_acm) == config['test_run']['experiment1']['num_epochs']
        assert len(acc_acm) == config['test_run']['experiment1']['num_epochs']
        
        print("Experiment 1 test passed!")
        return True
    except Exception as e:
        import traceback
        print(f"Experiment 1 test failed: {e}")
        print("Debug: Traceback:")
        traceback.print_exc()
        return False


def test_experiment2(config):
    """
    Test Experiment 2: Adaptive Behavior on Ill-Conditioned Synthetic Loss Landscapes.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        bool: True if the test passed, False otherwise.
    """
    print("\n=== Testing Experiment 2 ===")
    try:
        traj_acm, traj_sgd, traj_adam = experiment2(config, test_run=True)
        
        # Basic validation
        assert traj_acm.shape[0] == config['test_run']['experiment2']['num_steps'] + 1
        assert traj_acm.shape[1] == 2  # 2D parameters
        
        print("Experiment 2 test passed!")
        return True
    except Exception as e:
        print(f"Experiment 2 test failed: {e}")
        return False


def test_experiment3(config):
    """
    Test Experiment 3: Sensitivity Analysis of Adaptive Regularization and Curvature Scaling.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        bool: True if the test passed, False otherwise.
    """
    print("\n=== Testing Experiment 3 ===")
    try:
        study = experiment3(config, test_run=True)
        
        # Basic validation
        assert len(study.trials) == config['test_run']['experiment3']['n_trials']
        
        print("Experiment 3 test passed!")
        return True
    except Exception as e:
        print(f"Experiment 3 test failed: {e}")
        return False


def test_all():
    """
    Run all tests.
    
    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("+++ Starting test of all experiments +++")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up test configuration
    config = setup_test_config()
    
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
    
    # Run tests
    test1_passed = test_experiment1(config)
    test2_passed = test_experiment2(config)
    test3_passed = test_experiment3(config)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Experiment 1: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Experiment 2: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Experiment 3: {'PASSED' if test3_passed else 'FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nOverall: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_passed


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
