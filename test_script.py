"""
Simple test script to verify the STEM implementation.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.utils import prepare_logs_directory
from src.preprocess import initialize_models

def test_imports():
    """Test that all required modules can be imported."""
    print("PyTorch version:", torch.__version__)
    print("All imports successful!")
    
def test_model_initialization():
    """Test that the transformer models can be initialized."""
    print("Initializing models...")
    initialize_models()
    print("Models initialized successfully!")
    
def test_logs_directory():
    """Test that the logs directory can be created."""
    prepare_logs_directory()
    print("Logs directory prepared successfully!")

if __name__ == "__main__":
    print("Running simple tests for STEM implementation...")
    test_imports()
    test_logs_directory()
    print("All tests completed successfully!")
