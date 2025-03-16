"""
Test script to verify the ACM optimizer implementation works correctly.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.optimizers import ACM
from src.preprocess import set_seed, get_device, synthetic_function, generate_two_moons_data
from src.train import SimpleNet, run_synthetic_experiment, train_two_moons_model

def test_acm_optimizer():
    """
    Test the ACM optimizer on a simple optimization problem.
    """
    print("\n" + "="*50)
    print("Testing ACM Optimizer")
    print("="*50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create a simple tensor to optimize
    xy = torch.tensor([4.0, 4.0], requires_grad=True)
    
    # Initialize ACM optimizer
    optimizer = ACM([xy], lr=0.1, beta=0.9, curvature_scale=1.0)
    
    # Run a few optimization steps
    print("\nRunning optimization steps:")
    for step in range(10):
        optimizer.zero_grad()
        loss = synthetic_function(xy)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}: Loss = {loss.item():.4f}, Position = [{xy[0].item():.4f}, {xy[1].item():.4f}]")
    
    print("\nACM optimizer test completed successfully.")
    return True

def test_two_moons_classification():
    """
    Test the ACM optimizer on a simple two-moons classification problem.
    """
    print("\n" + "="*50)
    print("Testing Two-Moons Classification")
    print("="*50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Generate a small two-moons dataset
    print("\nGenerating two-moons dataset...")
    X_train, y_train, X_val, y_val = generate_two_moons_data(
        n_samples=200,  # Small dataset for quick testing
        noise=0.2,
        test_size=0.2,
        seed=42
    )
    
    # Train a simple model with ACM
    print("\nTraining model with ACM optimizer...")
    model, losses, accuracy = train_two_moons_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        optimizer_name='acm',
        num_epochs=20,  # Small number of epochs for quick testing
        lr=0.05,
        beta=0.9,
        curvature_scale=1.0,
        hidden_dim=10,
        log_dir='./logs',
        model_dir='./models'
    )
    
    print(f"\nFinal validation accuracy: {accuracy:.4f}")
    print("\nTwo-moons classification test completed successfully.")
    return True

def main():
    """
    Run all tests to verify the implementation works correctly.
    """
    print("\n" + "="*50)
    print("ACM Optimizer Implementation Test")
    print("="*50)
    
    # Create output directories
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    # Run tests
    tests_passed = 0
    
    if test_acm_optimizer():
        tests_passed += 1
    
    if test_two_moons_classification():
        tests_passed += 1
    
    # Print summary
    print("\n" + "="*50)
    print(f"Tests completed: {tests_passed}/2 passed")
    print("="*50)

if __name__ == '__main__':
    main()
