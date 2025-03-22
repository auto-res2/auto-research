"""
Configuration for SBDT (Stochastic Backdoor with Diffused Triggers) experiments.
"""

# General configuration
DEVICE = "cuda"  # Use GPU for computation
RANDOM_SEED = 42  # For reproducibility
TEST_RUN = True  # Set to False for full experiments

# Dataset configuration
DATASET_NAME = "CIFAR10"
DATASET_PATH = "./data"
SUBSET_SIZE = 1000  # Number of samples to use for test run
BATCH_SIZE = 32

# Model configuration
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3

# Diffusion parameters
DIFFUSION_STEPS_OPTIONS = [5, 10, 15]
BETA_SCHEDULES = [(0.0001, 0.02), (0.001, 0.05)]
DEFAULT_DIFFUSION_STEPS = 10
DEFAULT_BETA_START = 0.0001
DEFAULT_BETA_END = 0.02

# Training parameters
NUM_EPOCHS = 3  # For test run
LEARNING_RATE = 1e-3

# Evaluation parameters
SSIM_THRESHOLD = 0.7  # Threshold for good reconstruction quality
