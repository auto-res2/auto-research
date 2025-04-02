"""
Implementation of the Probabilistic Temporal Diffusion Animator (PTDA) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PTDAModel(nn.Module):
    """
    Probabilistic Temporal Diffusion Animator (PTDA) model.
    
    This model extends the diffusion-based human animation framework by incorporating
    a variational inference module that explicitly models uncertainty in both
    appearance and background dynamics.
    """
    def __init__(self, include_latent=True, latent_dim=256, hidden_dim=512, num_layers=4, dropout=0.1):
        """
        Initialize the PTDA model.
        
        Args:
            include_latent: Whether to include the latent uncertainty branch
            latent_dim: Dimension of the latent space
            hidden_dim: Hidden dimension for the model
            num_layers: Number of layers in the model
            dropout: Dropout rate
        """
        super(PTDAModel, self).__init__()
        self.include_latent = include_latent
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        if include_latent:
            self.feature_size_large = self._get_feature_size(256)
            self.feature_size_small = self._get_feature_size(64)
            
            self.fc_mu_large = nn.Linear(self.feature_size_large, latent_dim)
            self.fc_logvar_large = nn.Linear(self.feature_size_large, latent_dim)
            
            self.fc_mu_small = nn.Linear(self.feature_size_small, latent_dim)
            self.fc_logvar_small = nn.Linear(self.feature_size_small, latent_dim)
            
            self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def _get_feature_size(self, input_size=64):
        """
        Calculate the feature map size after encoder.
        
        Args:
            input_size: Size of the input image (assuming square)
            
        Returns:
            int: Size of the flattened feature map
        """
        x = torch.zeros(1, 3, input_size, input_size)
        
        x = self.encoder(x)
        
        return x.view(1, -1).size(1)
    
    def encode(self, x):
        """
        Encode the input.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tuple of (features, mu, logvar) if include_latent=True, otherwise features
        """
        features = self.encoder(x)
        
        if self.include_latent:
            flat_features = features.view(features.size(0), -1)
            feature_size = flat_features.size(1)
            
            if feature_size == self.feature_size_large:
                mu = self.fc_mu_large(flat_features)
                logvar = self.fc_logvar_large(flat_features)
            else:
                mu = self.fc_mu_small(flat_features)
                logvar = self.fc_logvar_small(flat_features)
            
            return features, mu, logvar
        else:
            return features
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, features, latent=None):
        """
        Decode the features.
        
        Args:
            features: Feature tensor
            latent: Optional latent vector for conditioning
            
        Returns:
            Reconstructed image
        """
        if self.include_latent and latent is not None:
            latent_proj = self.latent_proj(latent).unsqueeze(2).unsqueeze(3)
            latent_proj = latent_proj.expand(-1, -1, features.size(2), features.size(3))
            
            features = features + latent_proj[:, :features.size(1), :, :]
        
        return self.decoder(features)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tuple of (reconstruction, latent) if include_latent=True, otherwise reconstruction
        """
        if self.include_latent:
            features, mu, logvar = self.encode(x)
            latent = self.reparameterize(mu, logvar)
            reconstruction = self.decode(features, latent)
            return reconstruction, latent
        else:
            features = self.encode(x)
            reconstruction = self.decode(features)
            return reconstruction
    
    def compute_kl_loss(self, mu, logvar):
        """
        Compute KL divergence loss.
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
            
        Returns:
            KL divergence loss
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
    
    def generate_sequence(self, initial_frame, num_frames=10):
        """
        Generate a sequence of frames starting from the initial frame.
        
        Args:
            initial_frame: Initial frame tensor of shape [1, C, H, W]
            num_frames: Number of frames to generate
            
        Returns:
            List of generated frames
        """
        self.eval()
        generated_frames = [initial_frame]
        current_frame = initial_frame
        
        with torch.no_grad():
            for _ in range(num_frames - 1):
                if self.include_latent:
                    next_frame, _ = self(current_frame)
                else:
                    next_frame = self(current_frame)
                
                generated_frames.append(next_frame)
                current_frame = next_frame
        
        return generated_frames


class AblatedPTDAModel(PTDAModel):
    """
    Ablated version of the PTDA model without the latent uncertainty branch.
    """
    def __init__(self, hidden_dim=512, num_layers=4, dropout=0.1):
        """
        Initialize the ablated PTDA model.
        
        Args:
            hidden_dim: Hidden dimension for the model
            num_layers: Number of layers in the model
            dropout: Dropout rate
        """
        super(AblatedPTDAModel, self).__init__(
            include_latent=False,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )


class BaselineModel(nn.Module):
    """
    Baseline model (MagicAnimate) without the latent uncertainty branch.
    """
    def __init__(self, hidden_dim=512, num_layers=4, dropout=0.1):
        """
        Initialize the baseline model.
        
        Args:
            hidden_dim: Hidden dimension for the model
            num_layers: Number of layers in the model
            dropout: Dropout rate
        """
        super(BaselineModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Reconstructed image
        """
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction
    
    def generate_sequence(self, initial_frame, num_frames=10):
        """
        Generate a sequence of frames starting from the initial frame.
        
        Args:
            initial_frame: Initial frame tensor of shape [1, C, H, W]
            num_frames: Number of frames to generate
            
        Returns:
            List of generated frames
        """
        self.eval()
        generated_frames = [initial_frame]
        current_frame = initial_frame
        
        with torch.no_grad():
            for _ in range(num_frames - 1):
                next_frame = self(current_frame)
                generated_frames.append(next_frame)
                current_frame = next_frame
        
        return generated_frames
