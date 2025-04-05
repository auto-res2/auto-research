"""
Model definitions for SphericalShift Point Transformer (SSPT) and baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import (
    SphericalProjection, ShiftedSphericalWindowAttention, FixedWindowAttention,
    DualModalAttention, SphericalPositionalEncoding, RelativePositionalEncoding
)

class SSPTModel(nn.Module):
    """
    SphericalShift Point Transformer (SSPT) model for classification.
    
    This model implements the SSPT architecture described in the method,
    with spherical projection, shifted window attention, dual-modal attention,
    and spherical positional encoding.
    """
    def __init__(self, num_classes=40, input_channels=3, hidden_dim=64):
        super(SSPTModel, self).__init__()
        self.projection = SphericalProjection(in_channels=input_channels, out_channels=hidden_dim)
        self.pos_enc = SphericalPositionalEncoding(channels=hidden_dim)
        self.attention = ShiftedSphericalWindowAttention(channels=hidden_dim)
        self.dual_attention = DualModalAttention(channels=hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                ShiftedSphericalWindowAttention(channels=hidden_dim),
                DualModalAttention(channels=hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(3)
        ])
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.pos_enc(x)
        x = self.attention(x)
        x = self.dual_attention(x)
        
        for layer in self.transformer_layers:
            x = x + layer(x)  # Residual connection
        
        x = x.transpose(1, 2)  # (B, channels, N)
        x = self.pool(x)       # (B, channels, 1)
        x = x.squeeze(-1)      # (B, channels)
        
        x = self.classifier(x)
        return x

class PTv3Model(nn.Module):
    """
    Simplified Baseline Model emulating PTv3.
    
    This is a baseline model for comparison with SSPT.
    """
    def __init__(self, num_classes=40, input_channels=3, hidden_dim=64):
        super(PTv3Model, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                FixedWindowAttention(channels=hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(3)
        ])
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.attention_layers:
            x = x + layer(x)  # Residual connection
        
        x = x.transpose(1, 2)  # (B, channels, N)
        x = self.pool(x)       # (B, channels, 1)
        x = x.squeeze(-1)      # (B, channels)
        
        x = self.classifier(x)
        return x

class SSPTVariant(nn.Module):
    """
    SSPT Variant for ablation study.
    
    This model allows for turning on/off different components of the SSPT model
    to study their individual contributions.
    """
    def __init__(self, num_classes=40, input_channels=3, hidden_dim=64,
                 use_spherical_projection=True,
                 use_shifted_attention=True,
                 use_dual_attention=True,
                 use_spherical_pos_enc=True):
        super(SSPTVariant, self).__init__()
        self.use_spherical_projection = use_spherical_projection
        self.use_shifted_attention = use_shifted_attention
        self.use_dual_attention = use_dual_attention
        self.use_spherical_pos_enc = use_spherical_pos_enc
        
        if self.use_spherical_projection:
            self.projection = SphericalProjection(in_channels=input_channels, out_channels=hidden_dim)
        else:
            self.projection = nn.Linear(input_channels, hidden_dim)
            
        if self.use_spherical_pos_enc:
            self.pos_enc = SphericalPositionalEncoding(channels=hidden_dim)
        else:
            self.pos_enc = RelativePositionalEncoding(channels=hidden_dim)
            
        if self.use_shifted_attention:
            self.attention = ShiftedSphericalWindowAttention(channels=hidden_dim)
        else:
            self.attention = FixedWindowAttention(channels=hidden_dim)
            
        self.dual_attention = DualModalAttention(channels=hidden_dim, use_vector_cor=self.use_dual_attention)
        
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                ShiftedSphericalWindowAttention(channels=hidden_dim) if self.use_shifted_attention else FixedWindowAttention(channels=hidden_dim),
                DualModalAttention(channels=hidden_dim, use_vector_cor=self.use_dual_attention),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(2)
        ])
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        if self.use_spherical_projection:
            x = self.projection(x)
        else:
            B, N, _ = x.shape
            x = x.view(-1, 3)
            x = self.projection(x)
            x = x.view(B, N, -1)
            
        x = self.pos_enc(x)
        
        x = self.attention(x)
        
        x = self.dual_attention(x)
        
        for layer in self.transformer_layers:
            x = x + layer(x)  # Residual connection
            
        x = x.transpose(1, 2)  # (B, channels, N)
        x = self.pool(x)       # (B, channels, 1)
        x = x.squeeze(-1)      # (B, channels)
        
        x = self.classifier(x)
        return x
