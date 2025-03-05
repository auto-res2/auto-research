import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from preprocess import preprocess_data
from train import train_models
from evaluate import evaluate_results

def load_config():
    """
    Load configuration from config file or use default configuration.
    
    Returns:
        dict: Configuration parameters
    """
    config_path = os.path.join('config', 'experiment_config.json')
    
    # Check if config file exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        # Use default configuration
        config = {
            # General settings
            'quick_test': True,  # Set to True for a quick test run
            'generate_plots': True,
            'save_models': True,
            
            # Experiment selection
            'run_synthetic': True,
            'run_cifar10': True,
            'run_mnist': True,
            'run_rosenbrock': True,
            'run_ablation': True,
            
            # Synthetic experiment settings
            'synthetic_num_iters': 100,
            
            # CIFAR-10 experiment settings
            'cifar10_epochs': 10,
            'cifar10_lr': 0.01,
            'cifar10_batch_size': 128,
            'cifar10_test_batch_size': 100,
            
            # MNIST experiment settings
            'mnist_epochs': 5,
            'mnist_lr': 0.01,
            'mnist_batch_size': 64,
            'mnist_test_batch_size': 1000,
            
            # ACM optimizer default settings
            'acm_beta': 0.9,
            'acm_curvature_influence': 0.1
        }
        print("Using default configuration")
    
    return config

def save_results(results, evaluation, config):
    """
    Save experiment results and evaluation to files.
    
    Args:
        results (dict): Dictionary containing experiment results
        evaluation (dict): Dictionary containing evaluation metrics
        config (dict): Configuration parameters
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    results_file = os.path.join('logs', f'results_{timestamp}.json')
    
    # Convert torch tensors to lists for JSON serialization
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_tensors(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Create a simplified version of results for saving
    simplified_results = {}
    
    if 'synthetic' in results:
        simplified_results['synthetic'] = []
        for result in results['synthetic']:
            simplified_result = {
                'optimizer': result['optimizer'],
                'quadratic': {
                    'final_loss': result['quadratic']['final_loss'],
                    'training_time': result['quadratic']['training_time']
                }
            }
            if result['rosenbrock']['losses']:
                simplified_result['rosenbrock'] = {
                    'final_loss': result['rosenbrock']['final_loss'],
                    'training_time': result['rosenbrock']['training_time']
                }
            simplified_results['synthetic'].append(simplified_result)
    
    if 'cifar10' in results:
        simplified_results['cifar10'] = {}
        for opt, result in results['cifar10'].items():
            simplified_results['cifar10'][opt] = {
                'final_train_acc': result['final_train_acc'],
                'final_test_acc': result['final_test_acc'],
                'final_train_loss': result['final_train_loss'],
                'final_test_loss': result['final_test_loss'],
                'total_time': result['total_time']
            }
    
    if 'mnist' in results:
        simplified_results['mnist'] = {}
        for opt, result in results['mnist'].items():
            simplified_results['mnist'][opt] = {
                'final_train_acc': result['final_train_acc'],
                'final_test_acc': result['final_test_acc'],
                'final_train_loss': result['final_train_loss'],
                'final_test_loss': result['final_test_loss'],
                'total_time': result['total_time']
            }
            if 'config' in result:
                simplified_results['mnist'][opt]['config'] = result['config']
    
    # Save simplified results
    with open(results_file, 'w') as f:
        json.dump(convert_tensors(simplified_results), f, indent=2)
    
    # Save evaluation
    evaluation_file = os.path.join('logs', f'evaluation_{timestamp}.json')
    with open(evaluation_file, 'w') as f:
        json.dump(convert_tensors(evaluation), f, indent=2)
    
    # Save configuration
    config_file = os.path.join('logs', f'config_{timestamp}.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Evaluation saved to {evaluation_file}")
    print(f"Configuration saved to {config_file}")

def print_experiment_header():
    """Print a header for the experiment."""
    print("\n" + "=" * 80)
    print("ADAPTIVE CURVATURE MOMENTUM (ACM) OPTIMIZER EXPERIMENT")
    print("=" * 80)
    print("\nThis experiment compares the performance of the ACM optimizer against")
    print("established optimizers (Adam, SGD with momentum) on various tasks:")
    print("1. Synthetic Optimization Benchmark (Convex Quadratic and Rosenbrock functions)")
    print("2. Deep Neural Network Training on CIFAR-10 using a simple CNN")
    print("3. Ablation Study & Hyperparameter Sensitivity Analysis on MNIST")
    print("\n" + "=" * 80 + "\n")

def main():
    """Main function to run the experiment."""
    # Print experiment header
    print_experiment_header()
    
    # Load configuration
    config = load_config()
    
    # Print configuration
    print("\nExperiment Configuration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    # Check if this is a quick test
    if config['quick_test']:
        print("\nRunning in QUICK TEST mode (reduced iterations/epochs for testing)")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Preprocess data
    print("\n\n" + "=" * 50)
    print("STEP 1: PREPROCESSING DATA")
    print("=" * 50)
    data = preprocess_data(config)
    
    # Step 2: Train models
    print("\n\n" + "=" * 50)
    print("STEP 2: TRAINING MODELS")
    print("=" * 50)
    results = train_models(data, config)
    
    # Step 3: Evaluate results
    print("\n\n" + "=" * 50)
    print("STEP 3: EVALUATING RESULTS")
    print("=" * 50)
    evaluation = evaluate_results(results, config)
    
    # Save results
    save_results(results, evaluation, config)
    
    # Print conclusion
    print("\n\n" + "=" * 50)
    print("EXPERIMENT CONCLUSION")
    print("=" * 50)
    
    print("\nThe Adaptive Curvature Momentum (ACM) optimizer was compared against")
    print("Adam and SGD with momentum on various optimization tasks.")
    
    # Summarize key findings
    print("\nKey Findings:")
    
    if 'synthetic' in evaluation:
        print("\n1. Synthetic Optimization:")
        print(f"   - Quadratic function: Best optimizer was {evaluation['synthetic']['quadratic']['best_optimizer']}")
        if 'rosenbrock' in evaluation['synthetic']:
            print(f"   - Rosenbrock function: Best optimizer was {evaluation['synthetic']['rosenbrock']['best_optimizer']}")
    
    if 'cifar10' in evaluation:
        print("\n2. CIFAR-10 Classification:")
        print(f"   - Best test accuracy: {evaluation['cifar10']['best_test_acc']['optimizer']} ({evaluation['cifar10']['best_test_acc']['value']:.2f}%)")
        print(f"   - Fastest optimizer: {evaluation['cifar10']['fastest_optimizer']['optimizer']}")
        
        # Compare ACM vs Adam
        if 'ACM' in evaluation['cifar10']['test_accs'] and 'Adam' in evaluation['cifar10']['test_accs']:
            acm_acc = evaluation['cifar10']['test_accs']['ACM']
            adam_acc = evaluation['cifar10']['test_accs']['Adam']
            diff = acm_acc - adam_acc
            print(f"   - ACM vs Adam: {diff:.2f}% {'better' if diff > 0 else 'worse'}")
    
    if 'mnist' in evaluation:
        print("\n3. MNIST Classification (Ablation Study):")
        print(f"   - Best test accuracy: {evaluation['mnist']['best_test_acc']['optimizer']} ({evaluation['mnist']['best_test_acc']['value']:.2f}%)")
        
        # If ablation study was run, report best hyperparameters
        best_acm = evaluation['mnist']['best_test_acc']['optimizer']
        if best_acm.startswith('ACM_b'):
            for opt, result in results['mnist'].items():
                if opt == best_acm and 'config' in result:
                    print(f"   - Best ACM hyperparameters: beta={result['config']['beta']}, curvature_influence={result['config']['curvature_influence']}")
                    break
    
    print("\nPlease check the logs directory for detailed results and plots.")
    
    return results, evaluation

if __name__ == "__main__":
    main()
