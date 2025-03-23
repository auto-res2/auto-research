"""
Main script for running HBFG-SE3 experiments.
"""
import os
import argparse
import torch
import matplotlib.pyplot as plt

from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model, benchmark_methods, evaluate_quality_and_diversity, run_ablation_study

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run HBFG-SE3 experiments')
    parser.add_argument('--test', action='store_true', help='Run quick tests')
    parser.add_argument('--skip-train', action='store_true', help='Skip training')
    return parser.parse_args()

def test_all():
    """Run quick tests for all experiments."""
    print("\n===== Starting Sanity Tests for All Experiments =====")
    
    # Get configs and device
    experiment_config, model_config, device = preprocess_data()
    
    # Override config for quick tests
    experiment_config['experiment']['efficiency']['num_iterations'] = 10
    experiment_config['experiment']['quality_diversity']['num_samples'] = 10
    experiment_config['experiment']['ablation']['num_steps'] = 5
    
    # Train model (quick version)
    trained_model = train_model(experiment_config, model_config, device)
    
    # Run experiments
    benchmark_methods(trained_model, experiment_config, device)
    evaluate_quality_and_diversity(experiment_config)
    run_ablation_study(trained_model, experiment_config, device)
    
    print("===== Sanity Tests Completed Successfully =====\n")

def main():
    """Main entry point for running experiments."""
    args = parse_args()
    
    # Preprocess data and get configs
    print("Starting HBFG-SE3 experiments...")
    experiment_config, model_config, device = preprocess_data()
    
    if args.test:
        # Run quick tests
        test_all()
        return
    
    # Train model
    if not args.skip_train:
        trained_model = train_model(experiment_config, model_config, device)
    else:
        # Load pre-trained model
        from utils.diffusion import BootstrappedForceNetwork, HBFGSE3Diffuser
        force_net = BootstrappedForceNetwork(
            atom_features=model_config['model']['atom_features'],
            hidden_dim=model_config['model']['hidden_dim']
        ).to(device)
        
        if os.path.exists('models/force_predictor.pt'):
            force_net.load_state_dict(torch.load('models/force_predictor.pt'))
            print("Loaded pre-trained force predictor.")
        else:
            print("No pre-trained model found. Please run without --skip-train first.")
            return
        
        trained_model = {
            'force_net': force_net,
            'diffuser': HBFGSE3Diffuser(force_net, device=device)
        }
    
    # Run experiments based on config
    results = evaluate_model(trained_model, experiment_config, device)
    
    print("All experiments completed successfully!")
    return results

if __name__ == '__main__':
    main()
