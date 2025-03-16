import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from train import train_models
from evaluate import evaluate_models

def load_config(config_path='config/experiment_config.json'):
    """
    Load experiment configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Using default configuration.")
        return {
            'seed': 42,
            'quick_test': False,
            'synthetic': {
                'num_iters': 200,
                'optimizers': {
                    'ACM': {
                        'lr': 0.1,
                        'beta': 0.9,
                        'curvature_influence': 0.05
                    },
                    'Adam': {
                        'lr': 0.1
                    },
                    'SGD_mom': {
                        'lr': 0.1,
                        'momentum': 0.9
                    }
                }
            },
            'cifar10': {
                'num_epochs': 10,
                'batch_size': 128,
                'optimizers': {
                    'ACM': {
                        'lr': 0.01,
                        'beta': 0.9,
                        'curvature_influence': 0.1
                    },
                    'Adam': {
                        'lr': 0.001
                    },
                    'SGD_mom': {
                        'lr': 0.01,
                        'momentum': 0.9
                    }
                }
            },
            'mnist': {
                'num_epochs': 5,
                'batch_size': 128,
                'ablation_study': {
                    'lr_values': [0.001, 0.01, 0.1],
                    'beta_values': [0.8, 0.9, 0.95],
                    'curvature_values': [0.01, 0.1, 0.5]
                }
            }
        }

def setup_experiment(config):
    """
    Set up the experiment environment.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        None
    """
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('paper', exist_ok=True)
    
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch to deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("\nDevice:", "CUDA" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def run_experiment(config):
    """
    Run the complete experiment pipeline.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        dict: Experiment results
    """
    print("\n=== Starting Adaptive Curvature Momentum (ACM) Optimizer Experiment ===\n")
    
    # Setup experiment environment
    setup_experiment(config)
    
    # Check if this is a quick test
    quick_test = config.get('quick_test', False)
    if quick_test:
        print("Running in quick test mode with reduced iterations/epochs")
    
    # Step 1: Preprocess data
    print("\n--- Step 1: Preprocessing Data ---")
    data = preprocess_data(config)
    
    # Step 2: Train models
    print("\n--- Step 2: Training Models ---")
    training_results = train_models(data, config)
    
    # Step 3: Evaluate models
    print("\n--- Step 3: Evaluating Models ---")
    evaluation_results = evaluate_models(data, training_results, config)
    
    # Step 4: Summarize results
    print("\n--- Step 4: Experiment Summary ---")
    
    # Synthetic optimization summary
    print("\nSynthetic Optimization:")
    for problem, results in evaluation_results['synthetic'].items():
        print(f"\n{problem.capitalize()} Function:")
        for optimizer, result in results.items():
            print(f"  {optimizer}: Final Loss = {result['final_loss']:.6f}")
    
    # CIFAR-10 summary
    print("\nCIFAR-10 Classification:")
    for optimizer, result in evaluation_results['cifar10'].items():
        print(f"  {optimizer}: Accuracy = {result['final_accuracy']:.2f}%, Time = {result['training_time']:.2f}s")
    
    # MNIST ablation study summary
    print("\nMNIST Ablation Study:")
    best_config = max(evaluation_results['mnist'].items(), key=lambda x: x[1]['final_accuracy'])
    print(f"  Best Configuration: {best_config[0]}")
    print(f"  Learning Rate: {best_config[1]['lr']}")
    print(f"  Momentum Factor (β): {best_config[1]['beta']}")
    print(f"  Curvature Influence: {best_config[1]['curvature_influence']}")
    print(f"  Final Accuracy: {best_config[1]['final_accuracy']:.2f}%")
    
    print("\n=== Experiment Completed Successfully ===")
    
    return evaluation_results

def quick_test():
    """
    Run a quick test of the experiment with minimal iterations.
    
    Returns:
        None
    """
    print("Running quick test of the ACM optimizer experiment...")
    
    # Create a minimal configuration for quick testing
    config = {
        'seed': 42,
        'quick_test': True,
        'synthetic': {
            'num_iters': 10
        },
        'cifar10': {
            'num_epochs': 1,
            'batch_size': 128
        },
        'mnist': {
            'num_epochs': 1,
            'batch_size': 128
        }
    }
    
    # Run the experiment with quick test configuration
    run_experiment(config)

if __name__ == "__main__":
    # Check if this is a quick test run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        quick_test()
    else:
        # Load configuration
        config = load_config()
        
        # Run the experiment
        run_experiment(config)
