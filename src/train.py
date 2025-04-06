"""
ClusterCloak: Training Module

This module contains functions for training surrogate models and
implementing the ClusterCloak poisoning method.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def train_dummy_model(generator, optimizer, criterion, latent_vectors, num_epochs=2, batch_size=32, 
                    experiment_type="ClusterCloak", save_plots=True):
    """
    Train a dummy model to simulate fine-tuning with poisoning defense.
    
    Args:
        generator: The generator model to train
        optimizer: Optimizer for training
        criterion: Loss function
        latent_vectors: Batch of latent vectors for generation
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        experiment_type: Type of experiment ('ClusterCloak' or 'MetaCloak')
        save_plots: Whether to save plots of training curves
    
    Returns:
        Tuple of (training_losses, final_model)
    """
    training_losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        images_generated = generator(latent_vectors)
        loss = criterion(images_generated, torch.randn_like(images_generated))
        loss.backward()
        optimizer.step()
        
        training_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{num_epochs}: Loss {loss.item():.4f}")
    
    if save_plots and training_losses:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, num_epochs+1), training_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{experiment_type} Training Loss")
        plt.savefig(f"logs/training_loss_{experiment_type.lower()}.pdf", format='pdf', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved training loss plot as: logs/training_loss_{experiment_type.lower()}.pdf")
    
    return training_losses, generator

class SimpleGenerator(nn.Module):
    """Simple generator network to simulate diffusion model components."""
    def __init__(self, latent_dim=100, img_size=224):
        super(SimpleGenerator, self).__init__()
        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * img_size * img_size)
        )
    
    def forward(self, z):
        img = self.fc(z)
        img = img.view(-1, 3, self.img_size, self.img_size)
        return img
