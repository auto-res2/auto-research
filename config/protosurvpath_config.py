"""
Configuration for ProtoSurvPath experiments.
"""

GENE_INPUT_DIM = 500  # Dimension of gene expression data
IMAGE_CHANNELS = 3    # Number of channels in histology images
HIDDEN_DIM = 128      # Hidden dimension for encoders
NUM_PROTOTYPES = 3    # Number of prototypes for clustering

BATCH_SIZE = 8        # Batch size for training
LEARNING_RATE = 1e-3  # Learning rate for optimizer
NUM_EPOCHS = 20       # Number of training epochs
TEST_SIZE = 0.2       # Proportion of data for test set
RANDOM_SEED = 42      # Random seed for reproducibility

IMAGE_SIZE = (224, 224)  # Size of histology images

QUICK_TEST = False    # Flag for quick test with reduced dataset
