"""
Configuration parameters for the STEM experiments.
"""

MODEL_HIDDEN_DIM = 128
METADATA_DIM = 1

TRAIN_EPOCHS = 3  # Using lower epochs for quick demonstration
LEARNING_RATE = 1e-3

SYNTHETIC_SAMPLES = 10
METADATA_SAMPLES = 10
DOMAIN_SAMPLES = 20

RUN_CONTROLLED_SYNTHETIC = True
RUN_ABLATION_STUDY = True
RUN_DOMAIN_SHIFT = True

BATCH_SIZE = 4  # Smaller to accommodate GPU memory constraints
