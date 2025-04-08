"""
Configuration for NTEC-G experiments.
"""

DEVICE = 'cpu'  # Use 'cuda' to run on GPU
SEED = 42
SAVE_DIR = 'logs'

LATENT_DIM = 128
BATCH_SIZE = 64

CONVERGENCE_EPSILON = 1e-4
MAX_ITERATIONS = 50
PROBE_STEPS = 3

NUM_SAMPLES = 16
DIFFUSION_STEPS = 10

REG_VARIANTS = {
    "no_reg": 0.0,
    "moderate_reg": 0.1, 
    "strong_reg": 0.5
}
REG_NUM_SAMPLES = 50
REG_TARGET_RADIUS = 1.0
