"""
Configuration for Score-Aligned Step Distillation (SASD) experiments.
"""

# General configuration
RANDOM_SEED = 42
DEVICE = "cuda"  # Use CUDA for GPU acceleration

# Dataset configurations
DATASETS = {
    "cifar10": {
        "batch_size": 128,
        "image_size": 32,
    },
    "celeba": {
        "batch_size": 64,
        "image_size": 64,
    }
}

# Model configurations
MODEL_CONFIG = {
    "hidden_channels": 128,
    "num_res_blocks": 2,
    "attention_resolutions": [16],
    "dropout": 0.1,
    "channel_mult": (1, 2, 2, 2),
}

# Diffusion configurations
DIFFUSION_CONFIG = {
    "num_steps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
}

# SASD configurations
SASD_CONFIG = {
    "lambda_values": [0.0, 0.1, 0.5, 1.0],  # Weight values for dual-loss objective
    "step_configs": [5, 10, 25],  # Number of diffusion steps to test
    "learnable_schedule": True,  # Whether to use learnable schedule
}

# Training configurations
TRAIN_CONFIG = {
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "clip_grad_norm": 1.0,
    "save_interval": 1000,
}

# Experiment configurations
EXPERIMENT_CONFIG = {
    "experiment1": {  # Ablation Study on Dual-Loss Objective
        "lambda_values": [0.0, 0.1, 0.5, 1.0],
        "num_epochs": 10,
        "batch_size": 64,
    },
    "experiment2": {  # Learnable Schedule vs. Fixed Schedule
        "num_epochs": 10,
        "batch_size": 64,
    },
    "experiment3": {  # Step Efficiency and Robustness Across Datasets
        "num_epochs": 5,
        "batch_size": 64,
        "step_configs": [5, 10, 25],
        "datasets": ["cifar10", "celeba"],
    },
    "test_mode": {  # Quick test configuration with minimal resources
        "num_epochs": 1,
        "batch_size": 32,
        "lambda_values": [0.0, 0.1],
        "step_configs": [5, 10],
        "datasets": ["cifar10"],
    },
}
