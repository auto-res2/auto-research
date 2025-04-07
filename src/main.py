"""
Main script for running HBFN experiments.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.hbfn_config import common_config, exp1_config, exp2_config, exp3_config

from preprocess import prepare_tree_data, prepare_mnist_data
from train import train_topology_experiment
from evaluate import evaluate_sampling_efficiency, evaluate_loss_impact

def setup_environment():
    """Set up the environment for experiments."""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    np.random.seed(common_config["seed"])
    torch.manual_seed(common_config["seed"])
    
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(common_config["seed"])
    else:
        print("CUDA is not available. Using CPU.")
        common_config["device"] = "cpu"
    
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_experiment_1():
    """Run Experiment 1: Topological Structure Preservation."""
    print("\n" + "="*80)
    print("Experiment 1: Topological Structure Preservation on Synthetic Tree Data")
    print("="*80)
    
    config = {**common_config, **exp1_config}
    
    tree_graph, dataset, data_loader = prepare_tree_data(config)
    
    baseline_model, hyperbolic_model, baseline_loss, hyper_loss = train_topology_experiment(
        tree_graph, dataset, data_loader, config
    )
    
    torch.save(baseline_model.state_dict(), os.path.join('models', 'baseline_model_exp1.pt'))
    torch.save(hyperbolic_model.state_dict(), os.path.join('models', 'hyperbolic_model_exp1.pt'))
    
    print("Experiment 1 completed successfully.")

def run_experiment_2():
    """Run Experiment 2: Enhanced Sampling Efficiency and Fidelity."""
    print("\n" + "="*80)
    print("Experiment 2: Enhanced Sampling Efficiency and Fidelity using MNIST")
    print("="*80)
    
    config = {**common_config, **exp2_config}
    
    mnist_data, data_loader = prepare_mnist_data(config)
    
    fid_scores = evaluate_sampling_efficiency(config)
    
    print("Experiment 2 completed successfully.")

def run_experiment_3():
    """Run Experiment 3: Combined Loss Impact and Optimization Stability."""
    print("\n" + "="*80)
    print("Experiment 3: Combined Loss Impact and Optimization Stability")
    print("="*80)
    
    config = {**common_config, **exp3_config}
    
    tree_graph, dataset, data_loader = prepare_tree_data(config)
    
    standard_loss, hyper_loss = evaluate_loss_impact(dataset, data_loader, config)
    
    print("Experiment 3 completed successfully.")

def main():
    """Main function to run all experiments."""
    setup_environment()
    
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    
    print("\nAll experiments completed successfully.")
    print(f"Experiment ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults saved in the 'logs' directory.")
    print("Models saved in the 'models' directory.")

if __name__ == "__main__":
    main()
