"""
Training module for FahDiff model.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import networkx as nx
from tqdm import tqdm
import random
from preprocess import set_seed


class HyperbolicEncoder(nn.Module):
    """Encoder for mapping graphs to hyperbolic latent space."""
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        
        # Map to hyperbolic latent space parameters
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        
        return mean, logvar


class HyperbolicDecoder(nn.Module):
    """Decoder for mapping from hyperbolic latent space to graphs."""
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Decoder layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        
        # Reconstruct adjacency matrix
        adj_logits = self.fc3(h)
        
        return adj_logits


class ForceModule(nn.Module):
    """Module for computing adaptive force guidance in hyperbolic space."""
    def __init__(self, latent_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Force estimation layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)  # Output force vectors in latent space
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        
        # Compute force vectors
        forces = self.fc3(h)
        
        return forces


class FahDiff(nn.Module):
    """
    Force-Enhanced Adaptive Hyperbolic Diffusion (FahDiff) model.
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, 
                 use_adaptive_force=True, use_dynamic_schedule=True,
                 dropout=0.1, diffusion_temperature=1.0, curvature_param=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_adaptive_force = use_adaptive_force
        self.use_dynamic_schedule = use_dynamic_schedule
        self.diffusion_temperature = diffusion_temperature
        self.curvature_param = curvature_param
        
        # Model components
        self.encoder = HyperbolicEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = HyperbolicDecoder(latent_dim, hidden_dim, input_dim, dropout)
        
        if use_adaptive_force:
            self.force_module = ForceModule(latent_dim, hidden_dim, dropout)
        else:
            self.force_module = None
            
    def encode(self, x):
        """Encode input to latent representation."""
        mean, logvar = self.encoder(x)
        return mean, logvar
        
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for sampling from latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def apply_diffusion(self, z, t, force=None):
        """Apply diffusion process in the hyperbolic space."""
        # Base diffusion
        noise = torch.randn_like(z) * self.diffusion_temperature
        
        # Apply adaptive force if enabled
        if self.use_adaptive_force and force is not None:
            # Scale force by time step (larger effect early in diffusion)
            force_scale = 1.0 - t if self.use_dynamic_schedule else 0.5
            z = z + force * force_scale
            
        # Apply diffusion step based on hyperbolic geometry
        curvature_effect = torch.exp(-self.curvature_param * t)
        z = z * curvature_effect + noise * t
        
        return z
        
    def decode(self, z):
        """Decode latent representation to output."""
        return self.decoder(z)
        
    def forward(self, x, diffusion_steps=10):
        """Forward pass through the entire model."""
        # Encode to latent space
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        
        # Apply diffusion process
        if self.training:
            # During training, apply multistep diffusion
            for t in torch.linspace(0, 1, diffusion_steps):
                force = self.force_module(z) if self.use_adaptive_force and self.force_module is not None else None
                z = self.apply_diffusion(z, t, force)
        
        # Decode from latent space
        output = self.decode(z)
        
        return output, mean, logvar, z


def train_fahdiff(config, G):
    """
    Train FahDiff model on a graph.
    
    Args:
        config: Configuration dictionary
        G: NetworkX graph to train on
        
    Returns:
        Trained model and training history
    """
    set_seed(config["seed"])
    
    # Convert graph to adjacency matrix
    adj_matrix = nx.to_numpy_array(G)
    adj_tensor = torch.FloatTensor(adj_matrix)
    
    # Create model
    n_nodes = G.number_of_nodes()
    model = FahDiff(
        input_dim=n_nodes,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        use_adaptive_force=config["use_adaptive_force"],
        use_dynamic_schedule=config["use_dynamic_schedule"],
        dropout=config["dropout"],
        diffusion_temperature=config["diffusion_temperature"],
        curvature_param=config["curvature_param"]
    )
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    adj_tensor = adj_tensor.to(device)
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    history = {"loss": []}
    epochs = config["epochs"]
    
    # Short training loop for test runs
    if config.get("test_run", False):
        epochs = min(5, epochs)
        
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output, mean, logvar, z = model(adj_tensor)
        
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy_with_logits(output, adj_tensor)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + 0.01 * kl_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record history
        history["loss"].append(loss.item())
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
    return model, history
