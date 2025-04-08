import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.utils.hyperbolic import exp_map, log_map, hyperbolic_distance

class HGCNEncoder(nn.Module):
    """
    Hyperbolic Graph Convolutional Network Encoder.
    Maps graph structured data to hyperbolic latent space.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = exp_map(x)
        return x

class DiffusionModel(nn.Module):
    """
    Diffusion model for denoising in hyperbolic space with consistency loss.
    """
    def __init__(self, encoder, use_consistency_loss=False, consistency_weight=1.0):
        super(DiffusionModel, self).__init__()
        self.encoder = encoder
        self.use_consistency_loss = use_consistency_loss
        self.consistency_weight = consistency_weight
        self.decoder = nn.Sequential(
            nn.Linear(encoder.conv2.out_channels, encoder.conv2.out_channels),
            nn.ReLU(),
            nn.Linear(encoder.conv2.out_channels, encoder.conv2.out_channels)
        )

    def forward(self, data, noise_level):
        latent = self.encoder(data.x, data.edge_index)
        noise = torch.randn_like(latent) * noise_level
        noisy_latent = latent + noise  
        denoised = self.decoder(noisy_latent)
        rec_loss = F.mse_loss(denoised, latent)
        
        consistency_loss = 0.0
        if self.use_consistency_loss:
            noise2 = torch.randn_like(latent) * (noise_level * 1.5)
            noisy_latent2 = latent + noise2
            denoised2 = self.decoder(noisy_latent2)
            consistency_loss = F.mse_loss(denoised, denoised2)
            
        return rec_loss + self.consistency_weight * consistency_loss

class DiffusionModelAnisotropic(nn.Module):
    """
    Diffusion model with anisotropic denoising in hyperbolic space.
    Separately handles radial and angular components.
    """
    def __init__(self, encoder, use_consistency_loss=True, consistency_weight=0.5):
        super(DiffusionModelAnisotropic, self).__init__()
        self.encoder = encoder
        self.use_consistency_loss = use_consistency_loss
        self.consistency_weight = consistency_weight
        self.decoder = nn.Sequential(
            nn.Linear(encoder.conv2.out_channels, encoder.conv2.out_channels),
            nn.ReLU(),
            nn.Linear(encoder.conv2.out_channels, encoder.conv2.out_channels)
        )

    def forward(self, data, radial_noise_level, angular_noise_level):
        latent = self.encoder(data.x, data.edge_index)
        noisy_latent = self._anisotropic_denoise(latent, radial_noise_level, angular_noise_level)
        denoised = self.decoder(noisy_latent)
        rec_loss = F.mse_loss(denoised, latent)
        
        consistency_loss = 0.0
        if self.use_consistency_loss:
            noisy_latent2 = self._anisotropic_denoise(latent, radial_noise_level*1.2, angular_noise_level*1.2)
            denoised2 = self.decoder(noisy_latent2)
            consistency_loss = F.mse_loss(denoised, denoised2)
            
        return rec_loss + self.consistency_weight * consistency_loss
    
    def _anisotropic_denoise(self, latent, radial_noise_level, angular_noise_level):
        tangent = log_map(latent)
        norm = torch.norm(tangent, dim=-1, keepdim=True)
        direction = tangent / (norm + 1e-8)
        
        radial_noise = torch.randn_like(norm) * radial_noise_level
        angular_noise = torch.randn_like(direction) * angular_noise_level
        
        new_norm = norm + radial_noise
        new_direction = direction + angular_noise
        new_direction = new_direction / (torch.norm(new_direction, dim=-1, keepdim=True) + 1e-8)
        new_tangent = new_norm * new_direction
        
        denoised_latent = exp_map(new_tangent)
        return denoised_latent
