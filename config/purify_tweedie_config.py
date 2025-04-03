"""
Configuration file for the Purify-Tweedie++ experiment.
"""
import torch

DATASET = "CIFAR10"
BATCH_SIZE = 128
NUM_WORKERS = 4

MODEL_TYPE = "resnet18"
PRETRAINED = True
NUM_CLASSES = 10

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50

ENABLE_DOUBLE_TWEEDIE = True
ENABLE_CONSISTENCY_LOSS = True
ENABLE_ADAPTIVE_COV = True

ATTACK_METHODS = ["FGSM", "PGD", "CW"]
ATTACK_EPSILON = 8/255
ATTACK_ALPHA = 2/255
ATTACK_STEPS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = 0

SAVE_PDF = True
PDF_DPI = 300
PLOT_STYLE = "whitegrid"
