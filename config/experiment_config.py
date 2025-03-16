"""
Configuration for ACM optimizer experiments.

This file contains parameters for running the experiments:
1. Synthetic function benchmarking
2. CIFAR-10 image classification
3. Ablation studies
"""

# Configuration dictionary
config = {
    'synthetic': {
        # Number of iterations for synthetic function optimization
        'iterations': 1000,  # Reduced from 5000 for faster execution
        # Early stopping threshold
        'loss_threshold': 1e-4,
        # Learning rate
        'lr': 1e-3,
        # Beta1 parameter (momentum decay)
        'beta1': 0.9,
        # Beta2 parameter (squared gradient decay)
        'beta2': 0.999,
        # Curvature coefficient for ACM
        'curvature_coeff': 1e-2
    },
    
    'cifar10': {
        # Number of training epochs
        'epochs': 5,  # Reduced from 10 for faster execution
        # Batch size (adjusted for Tesla T4 16GB VRAM)
        'batch_size': 256,
        # Number of data loader workers
        'num_workers': 4,
        # Learning rate
        'lr': 1e-3,
        # Beta1 parameter (momentum decay)
        'beta1': 0.9,
        # Beta2 parameter (squared gradient decay)
        'beta2': 0.999,
        # Curvature coefficient for ACM
        'curvature_coeff': 1e-2,
        # Weight decay (L2 regularization)
        'weight_decay': 1e-4,
        # Use subset of data for quick testing
        'use_subset': False,
        # Number of batches to use in subset
        'subset_batches': 10
    },
    
    'ablation': {
        # Number of iterations for ablation studies
        'iterations': 1000,  # Reduced from 2000 for faster execution
        # Early stopping threshold
        'loss_threshold': 1e-5
    }
}
