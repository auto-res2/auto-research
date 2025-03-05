#!/usr/bin/env python
"""
Adaptive Curvature Momentum (ACM) Optimizer Experiment

This script runs experiments comparing the ACM optimizer against established optimizers:
1. Synthetic Optimization Benchmark (Convex Quadratic and Rosenbrock-like functions)
2. Deep Neural Network Training on CIFAR-10 using a simple CNN
3. Ablation Study & Hyperparameter Sensitivity Analysis on MNIST

The ACM optimizer adjusts per-parameter learning rates using a simple curvature-estimate
(the difference between successive gradients) and uses momentum buffering.
"""

import os
import argparse
import torch
import matplotlib
# Use Agg backend for matplotlib (non-interactive, for saving plots)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json

from preprocess import preprocess_data
from train import train_models
from evaluate import evaluate_models

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ACM optimizer experiments')
    parser.add_argument('--quick-test', action='store_true', 
                        help='Run a quick test with reduced iterations/epochs')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--skip-synthetic', action='store_true',
                        help='Skip synthetic optimization experiments')
    parser.add_argument('--skip-cifar10', action='store_true',
                        help='Skip CIFAR-10 experiments')
    parser.add_argument('--skip-mnist', action='store_true',
                        help='Skip MNIST ablation study')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a JSON file."""
    if config_path is None or not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_environment():
    """Set up the environment for experiments."""
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/plots', exist_ok=True)

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_experiment_summary(config):
    """Print a summary of the experiment configuration."""
    print_section_header("EXPERIMENT CONFIGURATION")
    
    print("Adaptive Curvature Momentum (ACM) Optimizer Experiment")
    print("\nExperiments:")
    print("1. Synthetic Optimization Benchmark (Convex Quadratic and Rosenbrock-like functions)")
    print("2. Deep Neural Network Training on CIFAR-10 using a simple CNN")
    print("3. Ablation Study & Hyperparameter Sensitivity Analysis on MNIST")
    
    print("\nOptimizers being compared:")
    print("- ACM (Adaptive Curvature Momentum)")
    print("- Adam")
    print("- SGD with momentum")
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    print("\nDevice:", "CUDA" if torch.cuda.is_available() else "CPU")
    print("\n")

def run_experiment(args, config):
    """Run the complete experiment."""
    # Set random seed for reproducibility
    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.get('seed', 42))
    
    # Print experiment summary
    print_experiment_summary(config)
    
    # Start timing the experiment
    start_time = time.time()
    
    # Step 1: Preprocess data
    print_section_header("DATA PREPROCESSING")
    data = preprocess_data(config)
    
    # Step 2: Train models
    print_section_header("MODEL TRAINING")
    training_results = train_models(data, config)
    
    # Step 3: Evaluate models
    print_section_header("MODEL EVALUATION")
    evaluation_results = evaluate_models(data)
    
    # Print experiment completion time
    total_time = time.time() - start_time
    print_section_header("EXPERIMENT COMPLETED")
    print(f"Total experiment time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Print final summary
    print_section_header("FINAL SUMMARY")
    
    # Synthetic optimization summary
    if not args.skip_synthetic and 'synthetic' in evaluation_results:
        print("\n--- Synthetic Optimization Summary ---")
        
        # Quadratic function
        print("\nQuadratic Function:")
        if evaluation_results['synthetic'] and 'quadratic' in evaluation_results['synthetic']:
            quadratic_metrics = evaluation_results['synthetic']['quadratic']['metrics']
            for name, metrics in quadratic_metrics.items():
                print(f"{name}:")
                print(f"  Distance to Optimum: {metrics['distance_to_optimum']:.6f}")
        
        # Rosenbrock function
        print("\nRosenbrock Function:")
        if evaluation_results['synthetic'] and 'rosenbrock' in evaluation_results['synthetic']:
            rosenbrock_metrics = evaluation_results['synthetic']['rosenbrock']['metrics']
            for name, metrics in rosenbrock_metrics.items():
                print(f"{name}:")
                print(f"  Distance to Optimum [1,1]: {metrics['distance_to_optimum']:.6f}")
    
    # CIFAR-10 summary
    if not args.skip_cifar10 and 'cifar10' in evaluation_results:
        print("\n--- CIFAR-10 Training Summary ---")
        if evaluation_results['cifar10'] and 'metrics' in evaluation_results['cifar10']:
            cifar10_metrics = evaluation_results['cifar10']['metrics']
            for name, metrics in cifar10_metrics.items():
                print(f"{name}:")
                print(f"  Final Accuracy: {evaluation_results['cifar10']['results'][name]['final_accuracy']:.2f}%")
                print(f"  Training Efficiency: {metrics['training_efficiency']:.4f} accuracy/second")
    
    # MNIST ablation study summary
    if not args.skip_mnist and 'mnist' in evaluation_results:
        print("\n--- MNIST Ablation Study Summary ---")
        if evaluation_results['mnist'] and 'best_params' in evaluation_results['mnist']:
            best_params = evaluation_results['mnist']['best_params']
            print("\nBest ACM Hyperparameters:")
            print(f"  Learning Rate: {best_params['lr']}")
            print(f"  Beta (Momentum): {best_params['beta']}")
            print(f"  Curvature Influence: {best_params['curvature_influence']}")
            
            if 'results' in evaluation_results['mnist']:
                best_config = evaluation_results['mnist']['best_config']
                print(f"  Final Accuracy: {evaluation_results['mnist']['results'][best_config]['final_accuracy']:.2f}%")
    
    # ACM vs. Traditional Optimizers Comparison
    print("\n--- ACM vs. Traditional Optimizers Comparison ---")
    
    # For synthetic optimization
    if not args.skip_synthetic and 'synthetic' in evaluation_results and evaluation_results['synthetic']:
        print("\nSynthetic Optimization:")
        if 'quadratic' in evaluation_results['synthetic'] and 'rosenbrock' in evaluation_results['synthetic']:
            # Get metrics
            q_metrics = evaluation_results['synthetic']['quadratic']['metrics']
            r_metrics = evaluation_results['synthetic']['rosenbrock']['metrics']
            
            # Compare distance to optimum
            if 'ACM' in q_metrics and 'Adam' in q_metrics:
                q_acm = q_metrics['ACM']['distance_to_optimum']
                q_adam = q_metrics['Adam']['distance_to_optimum']
                q_sgd = q_metrics['SGD_mom']['distance_to_optimum']
                
                print(f"  Quadratic Function - Distance to Optimum:")
                print(f"    ACM: {q_acm:.6f}")
                print(f"    Adam: {q_adam:.6f}")
                print(f"    SGD+Mom: {q_sgd:.6f}")
                
                if q_acm < q_adam and q_acm < q_sgd:
                    print("    Result: ACM performed best")
                elif q_adam < q_acm and q_adam < q_sgd:
                    print("    Result: Adam performed best")
                else:
                    print("    Result: SGD+Mom performed best")
            
            # Compare Rosenbrock results
            if 'ACM' in r_metrics and 'Adam' in r_metrics:
                r_acm = r_metrics['ACM']['distance_to_optimum']
                r_adam = r_metrics['Adam']['distance_to_optimum']
                r_sgd = r_metrics['SGD_mom']['distance_to_optimum']
                
                print(f"  Rosenbrock Function - Distance to Optimum:")
                print(f"    ACM: {r_acm:.6f}")
                print(f"    Adam: {r_adam:.6f}")
                print(f"    SGD+Mom: {r_sgd:.6f}")
                
                if r_acm < r_adam and r_acm < r_sgd:
                    print("    Result: ACM performed best")
                elif r_adam < r_acm and r_adam < r_sgd:
                    print("    Result: Adam performed best")
                else:
                    print("    Result: SGD+Mom performed best")
    
    # For CIFAR-10
    if not args.skip_cifar10 and 'cifar10' in evaluation_results and evaluation_results['cifar10']:
        print("\nCIFAR-10 Classification:")
        if 'results' in evaluation_results['cifar10']:
            results = evaluation_results['cifar10']['results']
            
            if 'ACM' in results and 'Adam' in results and 'SGD_mom' in results:
                acm_acc = results['ACM']['final_accuracy']
                adam_acc = results['Adam']['final_accuracy']
                sgd_acc = results['SGD_mom']['final_accuracy']
                
                print(f"  Final Test Accuracy:")
                print(f"    ACM: {acm_acc:.2f}%")
                print(f"    Adam: {adam_acc:.2f}%")
                print(f"    SGD+Mom: {sgd_acc:.2f}%")
                
                if acm_acc > adam_acc and acm_acc > sgd_acc:
                    print("    Result: ACM performed best")
                elif adam_acc > acm_acc and adam_acc > sgd_acc:
                    print("    Result: Adam performed best")
                else:
                    print("    Result: SGD+Mom performed best")
    
    # Conclusion
    print("\n--- Conclusion ---")
    print("The Adaptive Curvature Momentum (ACM) optimizer demonstrates the ability to")
    print("adapt to the local curvature of the loss landscape, potentially offering")
    print("advantages in optimization tasks. The experiments show how ACM compares")
    print("with traditional optimizers like Adam and SGD with momentum across different")
    print("types of problems, from simple synthetic functions to deep neural networks.")
    
    print("\nThe ablation study on MNIST provides insights into the sensitivity of ACM")
    print("to its hyperparameters, with the best configuration achieving competitive")
    print("performance compared to established optimizers.")
    
    print("\nFuture work could explore more complex architectures, different datasets,")
    print("and further refinements to the curvature estimation mechanism.")

def main():
    """Main function to run the experiment."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.quick_test:
        config['quick_test'] = True
    if args.seed:
        config['seed'] = args.seed
    
    # Set up environment
    setup_environment()
    
    # Run the experiment
    run_experiment(args, config)

if __name__ == "__main__":
    main()
