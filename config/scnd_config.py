"""
Configuration parameters for the Spatially-Constrained Normal Diffusion (SCND) experiment.
"""

BATCH_SIZE = 1
IMAGE_SIZE = 256
CHANNELS = 3

NUM_SAMPLES = 3

NUM_DIFFUSION_STEPS = 20

NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
SDS_ALPHA = 0.1  # Weighting factor for spatial alignment

TEST_STEPS = 3
TEST_EPOCHS = 2
TEST_IMAGE_SIZE = 128
