# src/test_models.py
import torch
import os
from src.utils.models import AmbientDiffusionModel, OneStepGenerator

def test_models():
    """
    Test the AmbientDiffusionModel and OneStepGenerator with a simple input.
    """
    print("\n========== Testing Model Architectures ==========")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test input
    x = torch.randn(2, 3, 32, 32).to(device)
    print(f"Input shape: {x.shape}")
    
    # Test AmbientDiffusionModel
    print("\nTesting AmbientDiffusionModel...")
    ambient_model = AmbientDiffusionModel().to(device)
    ambient_output = ambient_model(x, step=10)
    print(f"AmbientDiffusionModel output shape: {ambient_output.shape}")
    
    # Test OneStepGenerator
    print("\nTesting OneStepGenerator...")
    one_step_model = OneStepGenerator().to(device)
    one_step_output = one_step_model(x)
    print(f"OneStepGenerator output shape: {one_step_output.shape}")
    
    print("\nModel test successful!")
    return True

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    # Run the test
    test_models()
