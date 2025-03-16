"""
Simple test script to verify the ACM optimizer implementation.
"""

import torch
import torch.nn as nn
from src.optimizers import ACMOptimizer

def test_acm_optimizer():
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # Create a random input and target
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Create optimizer
    optimizer = ACMOptimizer(model.parameters(), lr=0.01, curvature_coef=0.1)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Train for a few iterations
    for i in range(5):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {i+1}, Loss: {loss.item():.6f}")
    
    print("ACM optimizer test completed successfully!")

if __name__ == "__main__":
    print("Testing ACM optimizer...")
    test_acm_optimizer()
