"""
Configuration parameters for the Progressive Brightness Distillation Diffusion experiment.
"""

SEED = 42
DEVICE = "cuda"  # Use CUDA for GPU acceleration

DATASET_NAME = "CIFAR10"
BATCH_SIZE = 8
DOWNLOAD = True

DIFFUSION_STEPS = 5
TEACHER_CHANNELS = 3
STUDENT_CHANNELS = 16

NUM_EPOCHS = 3
LEARNING_RATE = 1e-3

OUTPUT_DIR = "outputs"
LOGS_DIR = "logs"
MODELS_DIR = "models"
