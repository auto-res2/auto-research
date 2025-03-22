"""
Configuration for the TCPGS experiments
"""

# Device configuration
DEVICE = "cuda"  # For GPU usage, "cpu" for CPU usage

# Dataset configuration
DATASET = "CIFAR10"
BATCH_SIZE = 128

# Model parameters
DENOISE_STEPS = {
    "full": 100,
    "medium": 50,
    "short": 25
}

# Experiment parameters
NOISE_LEVELS = {
    "Gaussian": [0.1, 0.2],
    "SaltPepper": [0.05, 0.1]
}

# Experiment names and descriptions
EXPERIMENTS = {
    "robustness": "Robustness to Corrupted Data and Variable Noise Levels",
    "convergence": "Convergence Efficiency (Fewer Sampling Steps)",
    "ablation": "Analysis of Gradient Estimation via Tweedie Consistency Correction"
}
