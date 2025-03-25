import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Simple CNN block for feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=False):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x

class SIDModel(nn.Module):
    """
    Baseline SiD (Score-based Image Distillation) model.
    Uses three separate networks.
    Memory-optimized version with reduced layers.
    """
    def __init__(self, in_channels=3, feature_dim=64, hidden_dim=128, use_bn=False):
        super(SIDModel, self).__init__()
        # Score network for true data distribution - reduced to 2 layers
        self.f_phi = nn.Sequential(
            SimpleCNN(in_channels, hidden_dim, use_bn=use_bn),
            SimpleCNN(hidden_dim, feature_dim, use_bn=use_bn)
        )
        
        # Score network for generated distribution - reduced to 2 layers
        self.f_psi = nn.Sequential(
            SimpleCNN(in_channels, hidden_dim, use_bn=use_bn),
            SimpleCNN(hidden_dim, feature_dim, use_bn=use_bn)
        )
        
        # Generator network - reduced to 2 layers
        self.G_theta = nn.Sequential(
            SimpleCNN(feature_dim, hidden_dim, use_bn=use_bn),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        score_true = self.f_phi(x)
        score_fake = self.f_psi(x)
        gen = self.G_theta(score_fake)
        return gen, score_true, score_fake

class TwiSTModel(nn.Module):
    """
    TwiST-Distill model with shared twin-network architecture.
    Reduces memory consumption compared to SiD.
    Memory-optimized version with reduced layers.
    """
    def __init__(self, in_channels=3, feature_dim=64, hidden_dim=128, use_bn=False):
        super(TwiSTModel, self).__init__()
        # Shared network for feature extraction - reduced to 2 layers
        self.shared_net = nn.Sequential(
            SimpleCNN(in_channels, hidden_dim, use_bn=use_bn),
            SimpleCNN(hidden_dim, feature_dim, use_bn=use_bn)
        )
        
        # Generator network - reduced to 2 layers
        self.G_theta = nn.Sequential(
            SimpleCNN(feature_dim, hidden_dim, use_bn=use_bn),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Extract features using shared network
        latent = self.shared_net(x)
        # Generate output using generator network
        gen = self.G_theta(latent)
        return gen, latent

    def compute_double_tweedie_consistency_loss(self, clean_data, noisy_data, noise_std):
        """
        Compute Double-Tweedie consistency loss.
        
        Args:
            clean_data: Clean input data
            noisy_data: Noisy input data
            noise_std: Noise standard deviation
            
        Returns:
            Consistency loss
        """
        # Get latent representations
        _, latent_clean = self.forward(clean_data)
        _, latent_noisy = self.forward(noisy_data)
        
        # Compute L2 loss between clean and noisy latent representations
        consistency_loss = F.mse_loss(latent_clean, latent_noisy)
        
        return consistency_loss
