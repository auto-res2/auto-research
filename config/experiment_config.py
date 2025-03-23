"""
Configuration for the ACM optimizer experiments.
"""
import torch

# Experiment settings
QUICK_TEST = False  # Set to True for a quick test run with fewer iterations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default optimizer hyperparameters
OPTIMIZER_CONFIGS = {
    "ACM": {"lr": 0.01, "beta": 0.9, "curvature_influence": 0.1},
    "Adam": {"lr": 0.001},
    "SGD_mom": {"lr": 0.01}
}

# Synthetic optimization settings
SYNTHETIC_ITERS = 100 if not QUICK_TEST else 10

# Neural network training settings
CIFAR_EPOCHS = 10 if not QUICK_TEST else 1
MNIST_EPOCHS = 5 if not QUICK_TEST else 1
BATCH_SIZE = 128
NUM_WORKERS = 2

# Ablation study settings
ABLATION_PARAM = "curvature_influence"
ABLATION_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5] if not QUICK_TEST else [0.01, 0.1]

# Results directory
RESULTS_DIR = "results"
