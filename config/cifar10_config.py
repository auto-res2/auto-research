"""
Configuration for CIFAR10 experiment
"""

# Model parameters
MODEL_NAME = 'resnet18'
NUM_CLASSES = 10

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 0.01
MOMENTUM_BETA = 0.9
CURVATURE_INFLUENCE = 0.1
WEIGHT_DECAY = 5e-4
LR_STEP_SIZE = 20
LR_GAMMA = 0.1

# Quick test parameters
QUICK_TEST_EPOCHS = 2
