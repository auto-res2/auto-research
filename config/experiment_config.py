"""
Configuration for experiments
"""

# Synthetic experiment parameters
SYNTHETIC_CONFIG = {
    'quadratic': {
        'num_iters': 100,
    },
    'rosenbrock': {
        'num_iters': 200,
    }
}

# CIFAR-10 experiment parameters
CIFAR10_CONFIG = {
    'batch_size': 128,
    'test_batch_size': 100,
    'epochs': 10,
}

# MNIST ablation study parameters
MNIST_CONFIG = {
    'batch_size': 128,
    'test_batch_size': 1000,
    'epochs': 5,
    'lr_values': [0.001, 0.01, 0.1],
    'beta_values': [0.8, 0.9, 0.99],
    'curvature_values': [0.01, 0.05, 0.1],
}

# Quick test configuration (for verifying code execution)
QUICK_TEST_CONFIG = {
    'enabled': True,  # Set to False for full experiments
    'synthetic_iters': 10,
    'cifar10_epochs': 1,
    'mnist_epochs': 1,
}
