"""
Configuration file for the Adaptive Curvature Momentum (ACM) Optimizer experiments.

This file contains parameters for running the following experiments:
1. Synthetic Optimization Benchmark (Convex Quadratic and Rosenbrock-like functions)
2. Deep Neural Network Training on CIFAR-10 using a simple CNN
3. Ablation Study & Hyperparameter Sensitivity Analysis on MNIST
"""

# Configuration dictionary for all experiments
config = {
    # Data preprocessing parameters
    'synthetic': {
        'n_samples': 1000,  # Number of samples for synthetic data
        'seed': 42         # Random seed for reproducibility
    },
    'cifar10': {
        'batch_size': 128,  # Batch size for CIFAR-10 dataset
        'download': True    # Whether to download the dataset if not available
    },
    'mnist': {
        'batch_size': 128,  # Batch size for MNIST dataset
        'download': True    # Whether to download the dataset if not available
    },
    
    # Training parameters
    'train': {
        'seed': 42,                # Random seed for reproducibility
        'num_iters': 100,          # Number of iterations for synthetic optimization
        'num_epochs': 5,           # Number of epochs for neural network training
        
        # Optimizer parameters
        'lr': 0.01,                # Learning rate
        'beta': 0.9,               # Momentum factor
        'curvature_influence': 0.1, # Factor controlling the influence of curvature on learning rate
        'weight_decay': 0.0001,    # Weight decay (L2 penalty)
        
        'verbose': True            # Whether to print detailed training progress
    },
    
    # Ablation study parameters
    'ablation': {
        'seed': 42,                # Random seed for reproducibility
        'ablation_epochs': 3,      # Number of epochs for ablation study
        'ablation_batches': 100,   # Number of batches per epoch for ablation study
        
        # Default parameters
        'lr': 0.01,                # Default learning rate
        'beta': 0.9,               # Default momentum factor
        'curvature_influence': 0.1, # Default curvature influence factor
        
        # Parameter values to test
        'lr_values': [0.001, 0.01, 0.1],  # Learning rate values to test
        'beta_values': [0.8, 0.9, 0.95],  # Beta values to test
        'curvature_influence_values': [0.01, 0.1, 0.5]  # Curvature influence values to test
    },
    
    # Evaluation parameters
    'evaluate': {
        'seed': 42  # Random seed for reproducibility
    },
    
    # Quick test parameters (for verifying code execution)
    'quick_test': {
        'enabled': False,  # Whether to run in quick test mode
        'num_iters': 5,    # Number of iterations for synthetic optimization
        'num_epochs': 1,   # Number of epochs for neural network training
        'batch_limit': 5   # Number of batches to process in quick test mode
    }
}

# Function to get the configuration
def get_config():
    """
    Get the configuration dictionary.
    
    Returns:
        Dictionary containing configuration parameters
    """
    return config

# Function to get a quick test configuration
def get_quick_test_config():
    """
    Get a configuration dictionary for quick testing.
    
    Returns:
        Dictionary containing configuration parameters for quick testing
    """
    # Start with the full configuration
    test_config = get_config()
    
    # Modify parameters for quick testing
    test_config['synthetic']['n_samples'] = 10
    test_config['cifar10']['batch_size'] = 64
    test_config['mnist']['batch_size'] = 64
    
    test_config['train']['num_iters'] = 5
    test_config['train']['num_epochs'] = 1
    test_config['train']['verbose'] = False
    
    test_config['ablation']['ablation_epochs'] = 1
    test_config['ablation']['ablation_batches'] = 5
    test_config['ablation']['lr_values'] = [0.001, 0.01]
    test_config['ablation']['beta_values'] = [0.8, 0.9]
    test_config['ablation']['curvature_influence_values'] = [0.05, 0.1]
    
    test_config['quick_test']['enabled'] = True
    
    return test_config
