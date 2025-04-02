"""
Model definitions for LuminoDiff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BaseEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super(BaseEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 4, 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        return z


class DualLatentEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, 
                content_dim: int = 256, 
                brightness_dim: int = 64):
        super(DualLatentEncoder, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.content_conv = nn.Conv2d(128, content_dim, 4, 2, 1)
        self.brightness_conv = nn.Conv2d(128, brightness_dim, 4, 2, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(x)
        content_latent = self.content_conv(shared_features)
        brightness_latent = self.brightness_conv(shared_features)
        return content_latent, brightness_latent


class BrightnessBranch(nn.Module):
    def __init__(self, latent_dim: int, variant: str = 'A', weight: float = 1.0):
        super(BrightnessBranch, self).__init__()
        self.variant = variant
        self.weight = weight
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(self, z_brightness: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z_brightness.size()
        z_flat = z_brightness.view(b, -1)
        noise_pred = self.fc(z_flat)
        return noise_pred.view(b, c, h, w)


class AttentionFusion(nn.Module):
    def __init__(self, content_dim: int, brightness_dim: int, fusion_dim: int):
        super(AttentionFusion, self).__init__()
        self.content_proj = nn.Conv2d(content_dim, fusion_dim, 1)
        self.brightness_proj = nn.Conv2d(brightness_dim, fusion_dim, 1)
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4)
        
    def forward(self, content_latent: torch.Tensor, 
               brightness_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = content_latent.size()
        content_feat = self.content_proj(content_latent).view(b, -1, h * w).transpose(0, 1)
        brightness_feat = self.brightness_proj(brightness_latent).view(b, -1, h * w).transpose(0, 1)
        fused_feat, attn_weights = self.attention(brightness_feat, content_feat, content_feat)
        fused_feat = fused_feat.transpose(0, 1).view(b, -1, h, w)
        return fused_feat, attn_weights


class BaselineFusion(nn.Module):
    def __init__(self, content_dim: int, brightness_dim: int, fusion_dim: int):
        super(BaselineFusion, self).__init__()
        self.conv = nn.Conv2d(content_dim + brightness_dim, fusion_dim, kernel_size=1)
        
    def forward(self, content_latent: torch.Tensor, 
               brightness_latent: torch.Tensor) -> Tuple[torch.Tensor, None]:
        fused = torch.cat([content_latent, brightness_latent], dim=1)
        fused_feat = self.conv(fused)
        return fused_feat, None


class DualLatentFusionModel(nn.Module):
    def __init__(self, encoder: nn.Module, fusion_module: nn.Module):
        super(DualLatentFusionModel, self).__init__()
        self.encoder = encoder  # instance of DualLatentEncoder
        self.fusion = fusion_module
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        content_latent, brightness_latent = self.encoder(x)
        fused, attn_weights = self.fusion(content_latent, brightness_latent)
        return fused, attn_weights
