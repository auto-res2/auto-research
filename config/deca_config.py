
RANDOM_SEED = 42
IMAGE_SIZE = (64, 64)  # Base image size for experiments
DEVICE = "cuda"  # Use GPU by default

NUM_CLASSES = 4
IMAGES_PER_CLASS = 100
BATCH_SIZE = 32

DIFFUSION_STRENGTH_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
WEAK_CLASSES = [1, 3]  # Classes to enhance with diffusion
DIFFUSION_SAMPLES_PER_CLASS = 50

NUM_EPOCHS = 3  # For quick testing
LEARNING_RATE = 1e-3

RUN_EXPERIMENT_1 = True  # Classification Performance
RUN_EXPERIMENT_2 = True  # Diffusion Strength Ablation
RUN_EXPERIMENT_3 = True  # Curriculum Learning
RUN_TEST_ONLY = False  # Set to True for quick testing
