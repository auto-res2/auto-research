"""
Main script for running PurifyCov++ experiments.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from preprocess import preprocess_data
from train import load_models
from evaluate import experiment1, experiment2, experiment3
from utils.models import SimpleClassifier, CovarianceNet

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PurifyCov++ Experiments')
    parser.add_argument('--quick_test', action='store_true', 
                        help='Run a quick test with a small subset of data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loaders')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loaders')
    parser.add_argument('--exp1', action='store_true',
                        help='Run Experiment 1: Comparison under Varied Adversarial Attacks')
    parser.add_argument('--exp2', action='store_true',
                        help='Run Experiment 2: Ablation Study on the Covariance Prediction Module')
    parser.add_argument('--exp3', action='store_true',
                        help='Run Experiment 3: Efficiency and Convergence Analysis')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    return parser.parse_args()

def setup_environment():
    """Set up the environment for experiments."""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return device

def code_test():
    """
    A quick test function to verify that the code runs correctly.
    This loads a small subset of the CIFAR-10 test set (1 batch) and runs
    each experiment using a dummy classifier and covariance network.
    """
    print("=== Running Code Test ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_loader = preprocess_data(batch_size=8, quick_test=True)
    
    classifier = SimpleClassifier(num_classes=10).to(device)
    covariance_net = CovarianceNet(in_channels=3).to(device)
    
    classifier.eval()
    covariance_net.eval()
    
    results_exp1 = experiment1(device, test_loader, classifier, covariance_net)
    print("Experiment 1 results:", results_exp1)
    
    results_exp2 = experiment2(device, test_loader, classifier, covariance_net)
    print("Experiment 2 results:", results_exp2)
    
    results_exp3 = experiment3(device, test_loader, classifier, covariance_net)
    print("Experiment 3 results:", results_exp3)
    
    print("=== Code Test Completed ===")

def main():
    """Main function to run experiments."""
    args = parse_args()
    device = setup_environment()
    
    print("=== PurifyCov++: Covariance-Optimized Diffusion Purification ===")
    print("This method enhances diffusion purification by adaptively controlling")
    print("randomness with input-dependent, learned covariance estimation.")
    print()
    
    if args.quick_test:
        code_test()
        return
    
    test_loader = preprocess_data(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        quick_test=False
    )
    
    classifier, covariance_net = load_models(device)
    
    if args.all or args.exp1:
        print("\nRunning Experiment 1: Comparison under Varied Adversarial Attacks")
        results_exp1 = experiment1(device, test_loader, classifier, covariance_net)
        print("Experiment 1 results:", results_exp1)
    
    if args.all or args.exp2:
        print("\nRunning Experiment 2: Ablation Study on the Covariance Prediction Module")
        results_exp2 = experiment2(device, test_loader, classifier, covariance_net)
        print("Experiment 2 results:", results_exp2)
    
    if args.all or args.exp3:
        print("\nRunning Experiment 3: Efficiency and Convergence Analysis")
        results_exp3 = experiment3(device, test_loader, classifier, covariance_net)
        print("Experiment 3 results:", results_exp3)
    
    print("\n=== All experiments completed successfully ===")
    print("Results and figures have been saved to the logs directory.")

if __name__ == "__main__":
    main()
