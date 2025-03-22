"""
Simple U-Net like model for diffusion processes.
"""

import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    """
    A simple U-Net like model for diffusion-based image generation.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Autoencoder(nn.Module):
    """
    Simple autoencoder for anomaly detection.
    """
    def __init__(self, in_channels=3):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
