"""
Configuration for FahDiff experiments.
"""

# Base configuration for all experiments
BASE_CONFIG = {
    "seed": 42,
    "use_adaptive_force": True,
    "use_dynamic_schedule": True,
    "learning_rate": 1e-3,
    "epochs": 50,
    "diffusion_temperature": 1.0,
    "curvature_param": 0.5,
    "force_learning_rate": 1e-3,
    "dataset_sizes": [50, 100, 150],  # number of nodes for synthetic graphs
    "batch_size": 32,
    "hidden_dim": 64,
    "latent_dim": 32,
    "dropout": 0.1,
    "test_run": False,  # Set to True for quick test runs
}

# Configurations for hyperparameter sensitivity experiment
HYPERPARAM_GRID = {
    "diffusion_temperature": [0.5, 1.0, 1.5],
    "curvature_param": [0.1, 0.5, 1.0],
    "force_learning_rate": [1e-4, 1e-3, 1e-2]
}
