"""
Configuration file for the Probabilistic Temporal Diffusion Animator (PTDA) experiment.
"""

EXPERIMENT_NAME = "PTDA_Experiment"
RANDOM_SEED = 42
DEVICE = "cuda"  # Use CUDA for GPU acceleration

MODEL_CONFIG = {
    "include_latent": True,  # Whether to use the latent uncertainty branch
    "latent_dim": 256,       # Dimension of the latent space
    "hidden_dim": 512,       # Hidden dimension for the model
    "num_layers": 4,         # Number of layers in the model
    "dropout": 0.1,          # Dropout rate
}

TRAINING_CONFIG = {
    "batch_size": 4,         # Batch size for training
    "num_epochs": 50,        # Number of epochs for training
    "learning_rate": 1e-4,   # Learning rate
    "weight_decay": 1e-5,    # Weight decay for regularization
    "kl_weight": 0.01,       # Weight for the KL divergence loss
    "save_interval": 5,      # Save model every N epochs
}

DATA_CONFIG = {
    "num_frames": 10,        # Number of frames per sequence
    "frame_height": 256,     # Frame height
    "frame_width": 256,      # Frame width
    "num_channels": 3,       # Number of channels (RGB)
    "train_ratio": 0.8,      # Ratio of data to use for training
    "val_ratio": 0.1,        # Ratio of data to use for validation
    "test_ratio": 0.1,       # Ratio of data to use for testing
}

EXPERIMENT_CONFIG = {
    "experiment_1": {
        "name": "Dynamic Background Synthesis",
        "num_frames": 10,
        "metrics": ["ssim", "psnr", "temporal_consistency"],
    },
    "experiment_2": {
        "name": "Latent Variable Integration Ablation Study",
        "num_samples": 4,
        "num_frames": 5,
        "epochs": 2,
    },
    "experiment_3": {
        "name": "Long-Range Temporal Coherence Evaluation",
        "num_frames": 10,
        "window_size": 3,
    },
}

PATHS = {
    "data_dir": "data",
    "models_dir": "models",
    "logs_dir": "logs",
    "results_dir": "logs/results",
    "figures_dir": "logs/figures",
}
