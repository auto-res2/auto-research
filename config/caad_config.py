"""
Configuration for CAAD experiments.
"""

RANDOM_SEED = 42
DEVICE = 'cuda'  # Use GPU by default

DATASET_NAME = 'CIFAR10'
BATCH_SIZE = 64
NUM_WORKERS = 4

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10  # For full training, increase to 50+
TEST_EPOCHS = 3  # For quick testing

NOISE_ALPHA = 0.8  # Original image weight
NOISE_BETA = 0.2   # Structured noise weight
GAUSSIAN_KERNEL_SIZE = 7
GAUSSIAN_SIGMA = 2.0

DIFFUSION_ITERATIONS = 100
DIFFUSION_STEP_SIZE = 0.1
DIFFUSION_ERROR_THRESHOLD = 0.01

LIMITED_DATA_PERCENTAGES = [0.1, 0.2]  # 10% and 20% of the dataset

FIGURE_DPI = 300  # For high-quality PDF figures
