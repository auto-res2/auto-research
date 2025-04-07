"""
Configuration for Cov-Purify++ experiments.
"""

RANDOM_SEED = 42
DEVICE = "cuda"  # Use CUDA on T4 GPU

DATASET_NAME = "CIFAR10"
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 224  # For ResNet compatibility

MODEL_NAME = "resnet18"
PRETRAINED = True

ATTACK_TYPES = ["pgd", "bpda_eot", "black_box"]
EPSILON = 0.03  # Perturbation size

NUM_DIFFUSION_STEPS = 50
FIXED_LAMBDA = 0.5  # Fixed mixing parameter for Purify++

EXPERIMENT_TYPES = [
    "comparative_robustness",
    "adaptive_solver",
    "hyperparameter_sensitivity"
]

SAVE_DIR = "logs"
FIGURE_DPI = 300  # For high-quality PDF figures
