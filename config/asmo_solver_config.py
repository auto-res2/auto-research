"""
Configuration for ASMO-Solver experiments.
"""

RANDOM_SEED = 0
DEVICE = "cuda"  # Will use GPU if available
TEST_MODE = True  # Set to False for full experiment runs

EXP1_STEPS = [5, 10, 20] if not TEST_MODE else [5, 10]
EXP1_TIME_SPAN = (0., 1.)

EXP2_LATENT_DIM = 10
EXP2_TIMESTEPS = 100 if not TEST_MODE else 20
EXP2_WINDOW_SIZE = 20 if not TEST_MODE else 5
EXP2_MIN_DIM = 2
EXP2_MAX_DIM = 5
EXP2_THRESHOLD = 0.1

EXP3_LATENT_DIM = 2
EXP3_TIME_SPAN = (0., 1.)
EXP3_TEACHER_STEPS = 100 if not TEST_MODE else 20
EXP3_STUDENT_STEPS = 10
EXP3_TRAIN_EPOCHS = 50 if not TEST_MODE else 10
EXP3_LEARNING_RATE = 1e-3
