"""
ClusterCloak: Experiment Configuration

This file contains configuration parameters for ClusterCloak experiments.
"""

EXPERIMENT_NAME = "ClusterCloak"
RANDOM_SEED = 42
SAVE_PLOTS = True
DEVICE = "cuda"  # Use "cpu" for CPU-only running

NUM_SAMPLES = 500
BATCH_SIZE = 32
IMG_SIZE = 224

LATENT_DIM = 100
NUM_CLASSES = 10

NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

NOISE_EPSILON = 0.05
CLUSTERING_BIAS = 0.02

RUN_EXPERIMENT_1 = True  # Feature Misalignment & Clustering Analysis
RUN_EXPERIMENT_2 = True  # Fine-Tuning Recovery
RUN_EXPERIMENT_3 = True  # Robustness Under Transformations
