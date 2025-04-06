"""
Configuration parameters for the QD²P (Q-Denoising Diffusion Probe) experiment.
"""

RANDOM_SEED = 42
DEVICE = "cuda"  # Use "cpu" for CPU-only execution

LATENT_DIM = 16  # Dimension of latent vectors
BATCH_SIZE = 32  # Batch size for experiments

DIFFUSION_STEPS = 10  # Number of denoising steps
CANDIDATE_COUNT = 5  # Number of candidate proposals per step

CANDIDATE_COUNTS = [2, 5, 10]  # Different values of k to test
TEMPERATURES = [0.5, 1.0, 2.0]  # Different temperature values for softmax

SAVE_DIR = "logs"  # Directory to save plots and results
PDF_DPI = 300  # DPI for PDF figures
FIGURE_WIDTH = 10  # Width of figures in inches
FIGURE_HEIGHT = 6  # Height of figures in inches

TEST_LATENT_DIM = 8
TEST_STEPS = 3
TEST_CANDIDATE_COUNT = 3
