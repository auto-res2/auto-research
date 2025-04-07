"""
Model definitions for PurifyCov++ experiments.
"""

import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    """Simple CNN classifier for demonstration."""
    def __init__(self, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CovarianceNet(nn.Module):
    """Covariance prediction network for PurifyCov++."""
    def __init__(self, in_channels=3):
        super(CovarianceNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in (0,1): used as a multiplicative factor
        )
    
    def forward(self, x, t=None):
        return self.net(x)
