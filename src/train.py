import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.models import HGCNEncoder, DiffusionModel, DiffusionModelAnisotropic
from src.utils.hyperbolic import hyperbolic_distance

def train_experiment1(model, loader, optimizer, noise_level, epochs=1):
    """
    Train the diffusion model for Experiment 1: Impact of Consistency Loss.
    
    Args:
        model: DiffusionModel instance
        loader: DataLoader for the dataset
        optimizer: Optimizer for training
        noise_level: Level of noise to add
        epochs: Number of training epochs (default: 1)
        
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for data in loader:
            optimizer.zero_grad()
            loss = model(data, noise_level)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(loader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    
    return total_loss / epochs

def train_experiment2(model, loader, optimizer, radial_noise_level, angular_noise_level, epochs=1):
    """
    Train the anisotropic diffusion model for Experiment 2: Effects of Anisotropic Denoising.
    
    Args:
        model: DiffusionModelAnisotropic instance
        loader: DataLoader for the dataset
        optimizer: Optimizer for training
        radial_noise_level: Level of radial noise to add
        angular_noise_level: Level of angular noise to add
        epochs: Number of training epochs (default: 1)
        
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for data in loader:
            optimizer.zero_grad()
            loss = model(data, radial_noise_level, angular_noise_level)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(loader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    
    return total_loss / epochs
