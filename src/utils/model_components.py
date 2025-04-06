"""
Model components for ProtoSurvPath implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneExpressionEncoder(nn.Module):
    """Encoder for gene expression data."""
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the gene expression encoder.
        
        Args:
            input_dim: Dimension of input gene expression data
            hidden_dim: Dimension of hidden representations
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        """Forward pass through the encoder."""
        return self.encoder(x)


class VisionEncoder(nn.Module):
    """Encoder for histology image data."""
    
    def __init__(self, num_channels, hidden_dim):
        """
        Initialize the vision encoder.
        
        Args:
            num_channels: Number of input image channels
            hidden_dim: Dimension of hidden representations
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, hidden_dim)
        
    def forward(self, x):
        """Forward pass through the encoder."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module for multimodal integration."""
    
    def __init__(self, hidden_dim):
        """
        Initialize the cross-attention fusion module.
        
        Args:
            hidden_dim: Dimension of hidden representations
        """
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, gene_feat, image_feat):
        """
        Forward pass through the fusion module.
        
        Args:
            gene_feat: Features from gene expression encoder
            image_feat: Features from vision encoder
        """
        q = self.query(gene_feat).unsqueeze(1)  # [batch, 1, hidden_dim]
        k = self.key(image_feat).unsqueeze(1)   # [batch, 1, hidden_dim]
        v = self.value(image_feat).unsqueeze(1) # [batch, 1, hidden_dim]
        
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        context = torch.bmm(attn_weights, v)
        context = context.squeeze(1)
        
        fused = self.layer_norm(context + gene_feat)
        return fused


class SimpleFusion(nn.Module):
    """Simple concatenation fusion module (for ablation studies)."""
    
    def __init__(self, hidden_dim):
        """
        Initialize the simple fusion module.
        
        Args:
            hidden_dim: Dimension of hidden representations
        """
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, gene_feat, image_feat):
        """
        Forward pass through the fusion module.
        
        Args:
            gene_feat: Features from gene expression encoder
            image_feat: Features from vision encoder
        """
        fused = torch.cat([gene_feat, image_feat], dim=-1)
        return self.fc(fused)
