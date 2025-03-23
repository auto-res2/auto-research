"""
Model training for ABS-Diff experiments.

This script implements the training routines for the 
Adaptive Bayesian SDE-Guided Diffusion model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np

class RegimeClassifier(nn.Module):
    """
    Classifier to distinguish between memorization and generalization regimes.
    """
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, t):
        """
        Classify the input as memorization or generalization regime.
        
        Args:
            x: Input image
            t: Timestep
            
        Returns:
            scale: Scaling factor between 0.5 and 1.5
        """
        scale = 0.5 + self.classifier(x)
        return scale


class DiffusionModel(nn.Module):
    """
    Simplified diffusion model with optional regime classifier.
    """
    def __init__(self, regime_classifier=None):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
        self.regime_classifier = regime_classifier

    def forward(self, x, t):
        """
        Forward pass through the diffusion model.
        
        Args:
            x: Input image
            t: Timestep
            
        Returns:
            out: Output from the model
        """
        if self.regime_classifier is not None:
            regime_info = self.regime_classifier(x, t)
            # Ensure regime_info has the same spatial dimensions as unet output
            unet_out = self.unet(x)
            # Resize regime_info to match unet_out dimensions
            regime_info = torch.nn.functional.interpolate(
                regime_info, size=unet_out.shape[2:], mode='bilinear', align_corners=False
            )
            out = unet_out * regime_info
        else:
            out = self.unet(x)
        return out


class ABS_DiffModule(pl.LightningModule):
    """
    PyTorch Lightning module for training ABS-Diff.
    """
    def __init__(self, adaptive=True):
        super().__init__()
        self.regime_classifier = RegimeClassifier() if adaptive else None
        self.model = DiffusionModel(regime_classifier=self.regime_classifier)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, t):
        """
        Forward pass through the ABS-Diff module.
        
        Args:
            x: Input image
            t: Timestep
            
        Returns:
            Output from the model
        """
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        """
        Training step for the ABS-Diff module.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            loss: Training loss
        """
        x, _ = batch
        t = torch.randint(0, 1000, (x.size(0),), device=self.device).float()
        noise = torch.randn_like(x)
        x_noised = x + noise * (t.view(-1, 1, 1, 1) / 1000.0)
        noise_pred = self(x_noised, t)
        loss = self.loss_fn(noise_pred, noise)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        if batch_idx == 0:
            print(f"Batch {batch_idx}: loss = {loss.item():.6f}")
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers for training.
        
        Returns:
            optimizer: Adam optimizer
        """
        return optim.Adam(self.parameters(), lr=1e-3)


class DynamicDiffusionModel(nn.Module):
    """
    Diffusion model with dynamic noise conditioning.
    """
    def __init__(self, regime_classifier):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
        self.regime_classifier = regime_classifier

    def forward(self, x, t, noise_schedule):
        """
        Forward pass with dynamic noise conditioning.
        
        Args:
            x: Input image
            t: Timestep
            noise_schedule: Current noise schedule
            
        Returns:
            output: Model output
            dynamic_noise: Updated noise schedule
        """
        classifier_signal = self.regime_classifier(x, t)
        dynamic_noise = dynamic_noise_update(noise_schedule, classifier_signal)
        output = self.unet(x)
        
        # Reshape dynamic_noise to match output dimensions
        dynamic_noise_expanded = dynamic_noise.view(-1, 1, 1, 1).expand_as(output)
        
        return output * dynamic_noise_expanded, dynamic_noise


class AdaptiveSDESolver(nn.Module):
    """
    Adaptive SDE solver for different regimes.
    """
    def __init__(self, regime):
        super().__init__()
        self.regime = regime

    def forward(self, state, t):
        """
        SDE solver step.
        
        Args:
            state: Current state
            t: Timestep
            
        Returns:
            state_next: Next state
            noise_scale: Noise scale used
        """
        if self.regime == "memorization":
            noise_scale = 0.02  # stronger stochasticity to prevent overfitting
        else:
            noise_scale = 0.005  # lower noise for deterministic update
        noise = torch.randn_like(state) * noise_scale
        # A simple Euler–Maruyama update (dummy dynamics):
        state_next = state - 0.1 * state + noise
        return state_next, noise_scale


def dynamic_noise_update(current_noise, classifier_signal):
    """
    Update noise schedule dynamically based on classifier signal.
    
    Args:
        current_noise: Current noise schedule
        classifier_signal: Signal from regime classifier
        
    Returns:
        updated_noise: Updated noise schedule
    """
    updated_noise = current_noise * (1 - 0.1 * classifier_signal.mean())
    return updated_noise


def fixed_noise_schedule(timestep, total_steps=1000):
    """
    Fixed noise schedule based on timestep.
    
    Args:
        timestep: Current timestep
        total_steps: Total number of timesteps
        
    Returns:
        noise_level: Noise level for the timestep
    """
    return 1 - (timestep / total_steps)


def train_model(train_loader, num_epochs=1, adaptive=True):
    """
    Train the ABS-Diff model.
    
    Args:
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        adaptive: Whether to use adaptive regime classifier
        
    Returns:
        trained_module: Trained ABS-Diff module
    """
    module = ABS_DiffModule(adaptive=adaptive)
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False, 
        enable_checkpointing=False
    )
    trainer.fit(module, train_loader)
    return module
