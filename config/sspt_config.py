"""
Configuration file for SphericalShift Point Transformer experiments.
"""

RANDOM_SEED = 42
GPU_DEVICE = 0

NUM_POINTS = 1024  # Number of points per sample
NUM_CLASSES = 40   # Number of classes for ModelNet40

BATCH_SIZE = 16  # Adjusted for Tesla T4 (16GB VRAM)
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

USE_SPHERICAL_PROJECTION = True
USE_SHIFTED_ATTENTION = True
USE_DUAL_ATTENTION = True
USE_SPHERICAL_POS_ENC = True

RUN_ABLATION = True
RUN_ROBUSTNESS = True
