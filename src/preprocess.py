"""
RapidAlign: Preprocessing module

This module handles data preprocessing for RapidAlign experiments.
For the current experiments, preprocessing is minimal since we're using synthetic data.
"""

import numpy as np
import torch

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def preprocess():
    """Main preprocessing function, currently just sets seeds."""
    set_seeds()
