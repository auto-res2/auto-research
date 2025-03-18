"""
Configuration for transformer language modeling experiment
"""

# Model parameters
EMBED_SIZE = 200
NHEAD = 2
NHID = 200
NLAYERS = 2
DROPOUT = 0.2

# Training parameters
BATCH_SIZE = 20
BPTT = 35
NUM_EPOCHS = 15
LEARNING_RATE = 0.005
BETA = 0.9
CURVATURE_INFLUENCE = 0.1
LR_STEP_SIZE = 1
LR_GAMMA = 0.95

# Quick test parameters
QUICK_TEST_EPOCHS = 2
