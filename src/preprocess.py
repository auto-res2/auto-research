"""
Data preprocessing module for QD²P experiment.

For the synthetic experiment, this mainly involves creating synthetic data.
"""

import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_synthetic_data(latent_dim, batch_size, device="cuda"):
    """
    Create synthetic data for the QD²P experiment.
    
    Args:
        latent_dim: Dimension of latent vectors
        batch_size: Number of samples in batch
        device: Device to store tensors on
        
    Returns:
        Dictionary containing synthetic data
    """
    logger.info(f"Creating synthetic data: latent_dim={latent_dim}, batch_size={batch_size}")
    
    latent = torch.randn(batch_size, latent_dim, device=device)
    
    ideal = torch.ones((1, latent_dim), device=device)
    
    return {
        "latent": latent,
        "ideal": ideal
    }

def setup_experiment_directories():
    """Create necessary directories for the experiment."""
    dirs = ["logs", "data", "models", "config"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    logger.info("Created experiment directories")

if __name__ == "__main__":
    set_seed(42)
    setup_experiment_directories()
    data = create_synthetic_data(16, 4, device="cpu")
    print("Synthetic latent shape:", data["latent"].shape)
    print("Ideal latent shape:", data["ideal"].shape)
