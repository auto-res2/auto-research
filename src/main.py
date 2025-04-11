"""
Main script for running NSRPP experiments.

This script implements three experiments:
1. Comparison of NSRPP vs baseline bandit approach
2. Ablation study on surrogate model architectures
3. Evaluation of uncertainty-aware acquisition strategies
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yaml

np.random.seed(42)
torch.manual_seed(42)

from src.preprocess import load_config, prepare_experiment1_data, prepare_experiment2_data, prepare_experiment3_data
from src.train import run_nsrpp, run_baseline_bandit, run_experiment3_variant
from src.evaluate import evaluate_experiment1, evaluate_experiment2, evaluate_experiment3

def setup_environment():
    """
    Set up the environment for the experiments.
    
    Returns:
        dict: Configuration dictionary
    """
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/figures", exist_ok=True)
    
    config_path = "config/nsrpp/experiment_config.yaml"
    config = load_config(config_path)
    
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    return config

def run_experiment1(config):
    """
    Run Experiment 1: Compare NSRPP and Baseline Two-Level Bandit.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n[Experiment 1] Running NSRPP and Baseline Two-Level Bandit Experiment...")
    
    exp_params = prepare_experiment1_data(config)
    
    nsrpp_risk_history = run_nsrpp(
        num_iters=exp_params["num_iters"],
        delta=exp_params["delta"],
        learning_rate=exp_params["learning_rate"],
        init_theta=exp_params["init_theta"]
    )
    
    bandit_risk_history = run_baseline_bandit(
        num_iters=exp_params["num_iters"],
        delta=exp_params["delta"],
        learning_rate=exp_params["learning_rate"],
        init_theta=exp_params["init_theta"]
    )
    
    metrics = evaluate_experiment1(nsrpp_risk_history, bandit_risk_history, config["output_dir"])
    
    print("\nExperiment 1 Results:")
    print(f"NSRPP Final Risk: {metrics['nsrpp_final_risk']:.4f}")
    print(f"Bandit Final Risk: {metrics['bandit_final_risk']:.4f}")
    print(f"Risk Reduction: {metrics['risk_reduction_percent']:.2f}%")
    print(f"Sample Efficiency Improvement: {metrics['sample_efficiency_percent']:.2f}%")
    
    return metrics

def run_experiment2(config):
    """
    Run Experiment 2: Ablation Study on Surrogate Architectures.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n[Experiment 2] Running Ablation Study on Surrogate Architectures...")
    
    train_loader, val_dataset = prepare_experiment2_data(config)
    
    results = evaluate_experiment2(train_loader, val_dataset, config["output_dir"])
    
    print("\nExperiment 2 Results:")
    for name, mse in results.items():
        print(f"{name} Surrogate MSE: {mse:.4f}")
    
    return results

def run_experiment3(config):
    """
    Run Experiment 3: Uncertainty-Aware Acquisition Strategies.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n[Experiment 3] Running Uncertainty-Aware Acquisition Strategies Experiment...")
    
    exp_params = prepare_experiment3_data(config)
    
    ucb_history = run_experiment3_variant(
        num_iters=exp_params["num_iters"],
        learning_rate=exp_params["learning_rate"],
        init_theta=exp_params["init_theta"],
        use_ucb=True
    )
    
    pe_history = run_experiment3_variant(
        num_iters=exp_params["num_iters"],
        learning_rate=exp_params["learning_rate"],
        init_theta=exp_params["init_theta"],
        use_ucb=False
    )
    
    metrics = evaluate_experiment3(ucb_history, pe_history, config["output_dir"])
    
    print("\nExperiment 3 Results:")
    print(f"UCB Final Risk: {metrics['ucb_final_risk']:.4f}")
    print(f"Point Estimate Final Risk: {metrics['pe_final_risk']:.4f}")
    print(f"Risk Reduction with UCB: {metrics['risk_reduction_percent']:.2f}%")
    
    return metrics

def test_experiments(config):
    """
    Run minimal versions of the three experiments to check that everything executes.
    This test is designed to finish quickly.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("\n=== Starting Test of All Experiments ===")
    
    test_config = config.copy()
    test_config["experiment1"]["num_iters"] = 5
    test_config["experiment2"]["n_samples"] = 500
    test_config["experiment3"]["num_iters"] = 5
    
    print("\n[TEST] Experiment 1 (Quick Run)")
    run_experiment1(test_config)
    
    print("\n[TEST] Experiment 2 (Quick Run)")
    run_experiment2(test_config)
    
    print("\n[TEST] Experiment 3 (Quick Run)")
    run_experiment3(test_config)
    
    print("\n=== Test Finished Successfully ===")

def main():
    """
    Main function to run all experiments.
    """
    print("Starting NSRPP Experimental Suite")
    
    config = setup_environment()
    
    test_experiments(config)
    
    
    print("\nAll experiments completed successfully.")
    print("Results and figures saved in the logs directory.")

if __name__ == "__main__":
    main()
