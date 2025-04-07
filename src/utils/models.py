"""
Model components for Hyperbolic Bayesian Flow Networks.
"""

import torch
import torch.nn as nn
import numpy as np
import math

class BaselineEncoder(nn.Module):
    """Standard Euclidean encoder for BFN."""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(BaselineEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = torch.relu(self.fc(x))
        mu = self.latent_mu(h)
        logvar = self.latent_logvar(h)
        return mu, logvar

class HyperbolicEncoder(nn.Module):
    """Hyperbolic encoder for HBFN."""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(HyperbolicEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        self.manifold = None  # Placeholder for compatibility
    
    def forward(self, x):
        h = torch.relu(self.fc(x))
        mu = self.latent_mu(h)
        logvar = self.latent_logvar(h)
        norm = torch.norm(mu, dim=1, keepdim=True)
        scale = torch.tanh(norm) / (norm + 1e-8)
        hyper_mu = mu * scale
        return hyper_mu, logvar

class Decoder(nn.Module):
    """Decoder for reconstructing inputs from latent space."""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc(z))
        recon = self.out(h)
        return recon

class FlowNetwork(nn.Module):
    """Complete Flow Network combining encoder and decoder."""
    def __init__(self, encoder, decoder, device='cpu'):
        super(FlowNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, x):
        latent, logvar = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent, logvar
    
    def update(self, sample, t):
        """Simulated SDE update step. In a real implementation, this would
        use a more sophisticated solver considering radial/angular updates."""
        noise_scale = 0.01 * (1.0 - t / 30.0)  # Decreasing noise
        noise = torch.randn_like(sample) * noise_scale
        updated = sample - 0.01 * sample + noise
        return updated

class HyperbolicSDESolver:
    """SDE solver simulation for HBFN/SDE updates."""
    def __init__(self, model, num_iter=50, device='cpu'):
        self.model = model
        self.num_iter = num_iter
        self.device = device
    
    def sample(self, batch_size, latent_dim):
        """Generate samples by solving the SDE."""
        init_sample = torch.randn(batch_size, latent_dim).to(self.device)
        
        sample = init_sample
        for t in range(self.num_iter):
            sample = self.model.update(sample, t)
        
        return sample

def reconstruction_loss(recon, target):
    """Standard reconstruction loss."""
    return nn.MSELoss()(recon, target)

def hyperbolic_regulated_loss(latent_embeddings, expected_angles, expected_radii, manifold=None):
    """Hyperbolic-regulated loss component."""
    radii = torch.norm(latent_embeddings, dim=1)
    loss_radial = nn.MSELoss()(radii, expected_radii)
    
    if latent_embeddings.shape[1] >= 2:
        angles = torch.atan2(latent_embeddings[:, 1], latent_embeddings[:, 0])
        loss_angular = nn.MSELoss()(angles, expected_angles)
    else:
        loss_angular = 0.0
        
    return loss_radial + loss_angular
