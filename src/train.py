"""
Training functions for SBDT experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.diffusion_utils import forward_diffusion

def train_diffusion_model(model, dataloader, num_epochs=3, device="cpu", diffusion_fn=None, 
                         learning_rate=1e-3, save_path=None, test_run=True):
    """
    Train a diffusion model.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device to use for training
        diffusion_fn: Function to apply diffusion
        learning_rate: Learning rate for optimizer
        save_path: Path to save trained model
        test_run: Whether this is a test run (limits iterations)
        
    Returns:
        Trained model
    """
    if diffusion_fn is None:
        diffusion_fn = forward_diffusion
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    print("Starting training...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # Generate the noisy label representations
            noisy_labels = diffusion_fn(data)
            target = data  # For backdoor reconstruction, target is the original image
            
            optimizer.zero_grad()
            output = model(noisy_labels[-1])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # For test run, break early
            if test_run and i >= 3:
                break
                
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f}")
    
    # Save the trained model if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    return model

def train_autoencoder(autoencoder, dataloader, num_epochs=3, device="cpu", 
                     learning_rate=1e-3, save_path=None, test_run=True):
    """
    Train an autoencoder for anomaly detection.
    
    Args:
        autoencoder: Autoencoder model
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device to use for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save trained model
        test_run: Whether this is a test run (limits iterations)
        
    Returns:
        Trained autoencoder
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    autoencoder.train()
    
    print("Training Autoencoder on clean data...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon = autoencoder(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # For test run, break early
            if test_run and i >= 3:
                break
                
        print(f"Autoencoder Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f}")
    
    # Save the trained model if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(autoencoder.state_dict(), save_path)
        print(f"Autoencoder saved to {save_path}")
        
    return autoencoder
