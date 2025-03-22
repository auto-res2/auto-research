"""
Simple test script to verify the TCPGS implementation.
"""

import torch
import os
import sys

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.preprocess import get_dataset, add_gaussian_noise
from src.train import BaseDiffusionModel, TCPGSDiffusionModel
from config.tcpgs_config import BATCH_SIZE

def test_models():
    """Test the implementation of the diffusion models."""
    print("Testing TCPGS implementation...")
    
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model instances
    base_model = BaseDiffusionModel().to(device)
    tcpgs_model = TCPGSDiffusionModel(use_consistency=True).to(device)
    
    # Get a small batch of data
    print("Loading a small batch of CIFAR10 data...")
    train_loader = get_dataset(batch_size=4)  # Small batch size for testing
    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    # Add noise to images
    print("Adding noise to images...")
    noisy_images = add_gaussian_noise(images, std=0.2).to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        base_output = base_model(noisy_images)
        tcpgs_output = tcpgs_model(noisy_images)
    
    print(f"Base model output shape: {base_output.shape}")
    print(f"TCPGS model output shape: {tcpgs_output.shape}")
    
    # Test denoise_step
    print("Testing denoise_step...")
    with torch.no_grad():
        base_denoised = base_model.denoise_step(noisy_images, step=0, total_steps=10)
        tcpgs_denoised = tcpgs_model.denoise_step(noisy_images, step=0, total_steps=10)
    
    print(f"Base model denoised shape: {base_denoised.shape}")
    print(f"TCPGS model denoised shape: {tcpgs_denoised.shape}")
    
    # Test denoise_with_grad
    print("Testing denoise_with_grad...")
    base_denoised, base_grad = base_model.denoise_with_grad(noisy_images)
    tcpgs_denoised, tcpgs_grad = tcpgs_model.denoise_with_grad(noisy_images)
    
    print(f"Base model gradient shape: {base_grad.shape}")
    print(f"TCPGS model gradient shape: {tcpgs_grad.shape}")
    
    print("All tests passed successfully!")
    return True

if __name__ == "__main__":
    test_models()
