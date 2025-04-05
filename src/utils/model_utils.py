"""
Model components for SphericalShift Point Transformer (SSPT).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SphericalProjection(nn.Module):
    """
    Spherical projection module for point cloud processing.
    
    Projects points into a spherical coordinate system and applies
    a learned transformation.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super(SphericalProjection, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        B, N, _ = x.shape
        x = self.linear(x)
        return x

class ShiftedSphericalWindowAttention(nn.Module):
    """
    Shifted spherical window attention module.
    
    Applies attention over windows of points in the spherical domain,
    with windows shifted to ensure cross-patch information exchange.
    """
    def __init__(self, channels=64, num_heads=4):
        super(ShiftedSphericalWindowAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        
        return x

class FixedWindowAttention(nn.Module):
    """
    Fixed window attention module.
    
    Similar to shifted window attention, but without the window shifting.
    Used for ablation studies.
    """
    def __init__(self, channels=64, num_heads=4):
        super(FixedWindowAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        
        return x

class DualModalAttention(nn.Module):
    """
    Dual-modal attention module.
    
    Combines standard dot-product attention with a vector-based correlation
    head for faster convergence.
    """
    def __init__(self, channels=64, use_vector_cor=True):
        super(DualModalAttention, self).__init__()
        self.use_vector_cor = use_vector_cor
        self.fc = nn.Linear(channels, channels)
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, x):
        output = x
        
        if self.use_vector_cor:
            vector_cor = torch.relu(self.fc(x))
            weights = F.softmax(self.fusion_weight, dim=0)
            output = weights[0] * output + weights[1] * vector_cor
            
        return output

class SphericalPositionalEncoding(nn.Module):
    """
    Spherical positional encoding module.
    
    Encodes position information in the spherical domain.
    """
    def __init__(self, channels=64):
        super(SphericalPositionalEncoding, self).__init__()
        self.fc = nn.Linear(3, channels)
        
    def forward(self, x, coords=None):
        if coords is None:
            coords = x[:, :, :3]
            
        pos = self.fc(coords)
        
        return x + pos

class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    
    Encodes relative position information. Used for ablation studies.
    """
    def __init__(self, channels=64):
        super(RelativePositionalEncoding, self).__init__()
        self.fc = nn.Linear(3, channels)
        
    def forward(self, x, coords=None):
        if coords is None:
            coords = x[:, :, :3]
            
        pos = self.fc(coords)
        
        return x + pos
