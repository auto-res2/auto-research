"""Configuration for CSTD experiments."""

# Common parameters
RANDOM_SEED = 0
DEVICE = "cuda"  # For GPU usage

# Dataset parameters
DATASET = "cifar10"
BATCH_SIZE = 32
NUM_WORKERS = 4

# Trigger parameters
TRIGGER_PATCH_SIZE = 5
TRIGGER_IMPLANT_RATIO = 0.3

# Experiment 1: Ambient-Consistent Trigger Estimation parameters
EXP1_EPOCHS = 10
EXP1_LEARNING_RATE = 1e-3
EXP1_SIGMA1 = 0.1
EXP1_SIGMA2 = 0.05
EXP1_CONSISTENCY_WEIGHT = 0.5

# Experiment 2: Sequential Score-Based Trigger Refinement parameters
EXP2_EPOCHS = 10
EXP2_LEARNING_RATE = 1e-3
EXP2_NUM_STEPS = 10
EXP2_CONVERGENCE_THRESHOLD = 1e-4

# Experiment 3: Fast Defense Distillation parameters
EXP3_EPOCHS = 10
EXP3_LEARNING_RATE = 1e-3

# Testing parameters
TEST_MODE = False  # Set to True for quick test run
TEST_SUBSET_SIZE = 256
