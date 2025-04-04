"""
Model training for CAAD experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import os

class BasicDenoiser(nn.Module):
    """
    A simple convolutional denoiser network.
    """
    def __init__(self):
        super(BasicDenoiser, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.conv(x)

class CAADDenoiser(BasicDenoiser):
    """
    CAAD denoiser adds a covariance estimation branch.
    The output from the backbone is adjusted based on the covariance.
    """
    def __init__(self):
        super(CAADDenoiser, self).__init__()
        self.cov_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        denoised = super().forward(x)
        covariance = self.cov_branch(x)
        adjusted = denoised / (1 + torch.abs(covariance))
        return adjusted

def train_model(model, dataloader, optimizer, criterion=nn.MSELoss(), 
                epochs=5, device='cuda', add_noise_fn=None):
    """
    Generic training loop for reconstruction.
    
    Args:
        model: the model to train
        dataloader: data loader for the training data
        optimizer: the optimizer to use
        criterion: loss function (default: MSELoss)
        epochs: number of training epochs
        device: device to use for training
        add_noise_fn: function to add noise to input images
    
    Returns:
        trained model and list of training losses
    """
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            
            if add_noise_fn:
                corrupted = torch.stack([add_noise_fn(img) for img in data])
                corrupted = corrupted.to(device)
            else:
                corrupted = data
            
            optimizer.zero_grad()
            output = model(corrupted)
            loss = criterion(output, data)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f}")
    
    return model, losses

def train_with_validation(model, train_loader, val_loader, criterion=nn.MSELoss(),
                          epochs=10, device='cuda', add_noise_fn=None):
    """
    Train the model using a training loader and evaluate on a validation loader.
    
    Args:
        model: the model to train
        train_loader: data loader for training data
        val_loader: data loader for validation data
        criterion: loss function
        epochs: number of training epochs
        device: device to use for training
        add_noise_fn: function to add noise to input images
    
    Returns:
        trained model, training losses and validation losses
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        for data, _ in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            
            if add_noise_fn:
                corrupted = torch.stack([add_noise_fn(img) for img in data])
                corrupted = corrupted.to(device)
            else:
                corrupted = data
            
            optimizer.zero_grad()
            output = model(corrupted)
            loss = criterion(output, data)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{epochs}"):
                data = data.to(device)
                
                if add_noise_fn:
                    corrupted = torch.stack([add_noise_fn(img) for img in data])
                    corrupted = corrupted.to(device)
                else:
                    corrupted = data
                
                output = model(corrupted)
                loss = criterion(output, data)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")
    
    return model, train_losses, val_losses

def save_model(model, path):
    """Save model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, device='cuda'):
    """Load model from disk."""
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
