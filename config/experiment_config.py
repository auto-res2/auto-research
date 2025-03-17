"""
Configuration settings for the ACM optimizer experiments.
"""

# General configuration
RANDOM_SEED = 42
DEVICE = "cuda"  # Use GPU for training
TEST_MODE = False  # Set to True for quick test runs

# Experiment 1: Real-World Convergence (CIFAR-10 + ResNet-18)
REAL_WORLD_CONFIG = {
    "dataset": "cifar10",
    "batch_size": 128,
    "test_batch_size": 100,
    "num_workers": 2,
    "num_epochs": 30,
    "optimizers": {
        "ACM": {"lr": 0.1, "beta": 0.05, "weight_decay": 1e-4},
        "Adam": {"lr": 0.001, "weight_decay": 1e-4},
        "SGD": {"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4}
    },
    # For test mode
    "test_samples": 500,
    "test_epochs": 1
}

# Experiment 2: Synthetic Loss Landscape
SYNTHETIC_CONFIG = {
    "quadratic": {
        "iterations": 50,
        "alpha": 0.1,  # Base learning rate
        "beta": 0.05,  # Curvature influence factor
        "init_point": [3.0, 2.0],
        "matrix": [[3.0, 0.5], [0.5, 1.0]]
    },
    "rosenbrock": {
        "iterations": 100,
        "alpha": 0.1,
        "beta": 0.05,
        "init_point": [-1.5, 2.0],
        "a": 1.0,
        "b": 100.0
    },
    # For test mode
    "test_iterations_quadratic": 10,
    "test_iterations_rosenbrock": 10
}

# Experiment 3: Hyperparameter Sensitivity
HYPERPARAMETER_CONFIG = {
    "dataset": "cifar10",
    "batch_size": 128,
    "test_batch_size": 100,
    "num_workers": 2,
    "num_epochs": 20,
    "acm_grid": {
        "lr": [0.05, 0.1, 0.2],
        "beta": [0.01, 0.05, 0.1]
    },
    "adam_grid": {
        "lr": [0.0005, 0.001, 0.005]
    },
    # For test mode
    "test_samples": 1000,
    "test_epochs": 2
}
