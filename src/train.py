"""
Training module for GCAD experiments.

This module contains model definitions and training functions
for the Geometrically Consistent Ambient Diffusion experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import torch.nn.functional as F
from preprocess import corrupt_images

class PoincareBall:
    def __init__(self, c=1.0):
        self.c = c
    
    def expmap0(self, x):
        """
        Exponential map from origin in Euclidean space to Poincare ball
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-10)
        return torch.tanh(norm) * x / norm


class BaseAutoencoder(nn.Module):
    def __init__(self):
        super(BaseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

class GCADModel(nn.Module):
    def __init__(self):
        super(GCADModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

class GCADModelAblation(GCADModel):
    def __init__(self):
        super(GCADModelAblation, self).__init__()
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

class GCADHyperbolicEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(GCADHyperbolicEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, latent_dim, 3, stride=2, padding=1)
        )
        self.manifold = PoincareBall(c=1.0)
    
    def forward(self, x):
        euclidean_latent = self.encoder(x)
        euclidean_latent = euclidean_latent.view(x.size(0), -1)
        hyperbolic_latent = self.manifold.expmap0(euclidean_latent)
        return hyperbolic_latent


def train_model(model, data_loader, device, noise_type="gaussian", num_epochs=10, record_loss=False):
    """
    Train the given model with a specified corruption type.
    Returns the final model and a list of epoch losses (if record_loss is True).
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epoch_losses = []
    
    print(f"Training on noise type: {noise_type} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for images, _ in data_loader:
            images = images.to(device)
            corrupted = corrupt_images(images, noise_type=noise_type, noise_level=0.1)
            outputs = model(corrupted)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss_avg = epoch_loss / len(data_loader)
        epoch_losses.append(epoch_loss_avg)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss_avg:.4f} | Time={elapsed:.2f}s")
    
    return model, (epoch_losses if record_loss else None)

def extract_latents(model, data_loader, device, noise_type, encoder_fn):
    """
    Extract latent features from the given model using the provided encoder function.
    Returns numpy arrays of latent vectors and corresponding labels.
    """
    model.eval()
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            corrupted = corrupt_images(images, noise_type=noise_type, noise_level=0.1)
            latent = encoder_fn(corrupted)
            all_latents.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())
    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_latents, all_labels


def compute_psnr(mse, max_pixel=1.0):
    """Compute Peak Signal-to-Noise Ratio given MSE (using numpy log10)"""
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse)
