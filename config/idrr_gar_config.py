"""
Configuration for IDRR-GAR experiments.
"""

EXPERIMENT_NAME = "IDRR-GAR"
RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 1  # For demonstration; use larger value for full training
LEARNING_RATE = 1e-3

ITERATIONS = 3  # Number of refinement iterations
INPUT_CHANNELS = 3
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 320

DEVICE = "cuda"  # Use "cuda" for GPU, "cpu" for CPU
USE_FP16 = True  # Use half-precision for memory efficiency on Tesla T4

NUM_REGIONS = 50  # Number of regions to sample for geometry-aware sampling

PHOTOMETRIC_LOSS_WEIGHT = 1.0
SMOOTHNESS_LOSS_WEIGHT = 0.1
SCALE_ALIGNMENT_LOSS_WEIGHT = 0.1

DATA_DIR = "data/"
MODELS_DIR = "models/"
LOGS_DIR = "logs/"
