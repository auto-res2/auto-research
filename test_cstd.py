"""Test script for CSTD experiments."""

import os
import sys
import torch

# Add the repository root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from config import cstd_config as cfg
from src.main import test_experiments

if __name__ == "__main__":
    # Set test mode to True for quick run
    cfg.TEST_MODE = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run quick tests
    test_experiments(device)
