"""
Training script for HBFG-SE3 method.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm

from utils.diffusion import ForcePredictor, BootstrappedForceNetwork, HBFGSE3Diffuser
from utils.se3_utils import create_protein_batch, random_rotation_matrix

def train_force_predictor(config, device):
    """Train the force predictor network."""
    print("Training force predictor network...")
    
    # Create network
    force_net = BootstrappedForceNetwork(
        atom_features=config['model']['atom_features'],
        hidden_dim=config['model']['hidden_dim']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(force_net.parameters(), lr=1e-4)
    
    # Training parameters
    num_epochs = 50  # Small number for quick test
    batch_size = 8
    num_atoms = 10
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Create synthetic batch
        x = create_protein_batch(batch_size, num_atoms, device)
        
        # Generate random rotation matrices for augmentation
        rot_matrices = random_rotation_matrix(batch_size, device)
        
        # Apply rotations to maintain SE(3) equivariance
        x_rotated = torch.matmul(x, rot_matrices.transpose(1, 2))
        
        # Forward pass
        forces = force_net(x)
        forces_rotated = force_net(x_rotated)
        
        # Rotate forces back for consistency check (SE(3) equivariance)
        forces_rotated_back = torch.matmul(forces_rotated, rot_matrices)
        
        # Equivariance loss (forces should transform like vectors)
        equivariance_loss = nn.MSELoss()(forces, forces_rotated_back)
        
        # Energy minimization hint (simple regularization)
        energy_hint = torch.mean(torch.sum(forces**2, dim=-1))
        
        # Total loss
        loss = equivariance_loss + 0.01 * energy_hint
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(force_net.state_dict(), 'models/force_predictor.pt')
    
    return force_net

def train_model(experiment_config, model_config, device):
    """Train the HBFG-SE3 model."""
    print("Training HBFG-SE3 model...")
    
    # Train or load force predictor
    if os.path.exists('models/force_predictor.pt'):
        force_net = BootstrappedForceNetwork(
            atom_features=model_config['model']['atom_features'],
            hidden_dim=model_config['model']['hidden_dim']
        ).to(device)
        force_net.load_state_dict(torch.load('models/force_predictor.pt'))
        print("Loaded pre-trained force predictor.")
    else:
        force_net = train_force_predictor(model_config, device)
    
    # Create diffuser
    diffuser = HBFGSE3Diffuser(force_net, device=device)
    
    # Return the trained model components
    return {
        'force_net': force_net,
        'diffuser': diffuser
    }
