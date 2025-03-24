# src/utils/models.py
import torch
import torch.nn as nn

class AmbientDiffusionModel(nn.Module):
    """
    Ambient Diffusion Model for denoising diffusion.
    This model is trained on noisy data using the double Tweedie formula and consistency loss.
    """
    def __init__(self, channels=3, base_channels=32):
        super(AmbientDiffusionModel, self).__init__()
        # Simplified model architecture without skip connections to avoid dimension issues
        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            
            # Middle
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            
            # Decoder
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, channels, 3, padding=1)
        )
        
    def forward(self, x, step=None):
        # Add noise level conditioning if step is provided
        noise_level = 1.0 if step is None else 1.0 - (step / 50.0)
        
        # Apply model
        output = self.model(x)
        
        # Apply noise level conditioning
        return output * noise_level

class OneStepGenerator(nn.Module):
    """
    One-step generator implemented via Ambient Score Distillation (ASD).
    This model directly generates samples without requiring iterative denoising.
    """
    def __init__(self, channels=3, base_channels=32):
        super(OneStepGenerator, self).__init__()
        # Simplified encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, channels, 3, padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
