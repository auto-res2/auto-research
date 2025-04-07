"""
Configuration for Hyperbolic Bayesian Flow Networks experiments.
"""

common_config = {
    "seed": 42,
    "device": "cuda",  # Use GPU
    "num_workers": 4,
}

exp1_config = {
    "num_nodes": 50,
    "feature_dim": 10,
    "hidden_dim": 16,
    "latent_dim": 2,  # 2D for visualization
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "batch_size": 16,
}

exp2_config = {
    "input_dim": 28*28,
    "hidden_dim": 128,
    "latent_dim": 32,
    "learning_rate": 1e-3,
    "num_epochs": 10,
    "batch_size": 64,
    "iterations_to_test": [5, 10, 20, 30],
}

exp3_config = {
    "num_nodes": 50,
    "feature_dim": 10,
    "hidden_dim": 16,
    "latent_dim": 2,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "batch_size": 16,
    "hyperbolic_loss_weight": 0.1,
}
