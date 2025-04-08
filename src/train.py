"""
Training script for the ADNLCC (Ambient Diffusion with Non-Linear Characteristic Correction) method.

This implements a diffusion model with a special non-linear characteristic correction term
that improves training on noisy data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

class SimpleDiffusionModel(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(SimpleDiffusionModel, self).__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )
        
        self.noise_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
    def forward(self, x, noise_level=None):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        
        dec3 = self.dec3(enc3)
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))
        
        return dec1

class ADNLCCTrainer:
    def __init__(self, model, device, config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 1e-4)
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config.get('lr_step_size', 30),
            gamma=self.config.get('lr_gamma', 0.5)
        )
        
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        self.logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.training_log = {'epoch': [], 'loss': [], 'correction_norm': []}
        
    def compute_correction_term(self, pred, target, noise_level):
        """Compute the non-linear characteristic correction term for ADNLCC."""
        delta_x = torch.abs(pred - target).mean(dim=[1, 2, 3], keepdim=True)
        correction = delta_x * (1.0 - noise_level.view(-1, 1, 1, 1))
        return correction
    
    def train_epoch(self, dataloader, epoch, noise_levels, use_adnlcc=True):
        """Train for one epoch with the ADNLCC method."""
        self.model.train()
        epoch_loss = 0.0
        epoch_correction_norm = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)
            batch_size = data.shape[0]
            
            noise_level = noise_levels[batch_idx % len(noise_levels)]
            noise = torch.randn_like(data) * noise_level
            noisy_data = data + noise
            
            self.optimizer.zero_grad()
            pred = self.model(noisy_data, noise_level)
            
            base_loss = F.mse_loss(pred, data)
            
            if use_adnlcc:
                correction = self.compute_correction_term(pred, data, torch.tensor([noise_level], device=self.device))
                corrected_pred = pred - 0.1 * correction
                consistency_loss = F.mse_loss(corrected_pred, data)
                loss = base_loss + 0.5 * consistency_loss
                correction_norm = correction.mean().item()
            else:
                loss = base_loss
                correction_norm = 0.0
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_correction_norm += correction_norm
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'corr_norm': f"{correction_norm:.4f}"
            })
        
        self.scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_correction_norm = epoch_correction_norm / num_batches
        
        self.training_log['epoch'].append(epoch)
        self.training_log['loss'].append(avg_loss)
        self.training_log['correction_norm'].append(avg_correction_norm)
        
        return avg_loss, avg_correction_norm
    
    def save_model(self, epoch, filename=None):
        """Save the model checkpoint."""
        if filename is None:
            filename = f"adnlcc_model_epoch_{epoch}.pt"
        path = os.path.join(self.model_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_log': self.training_log
        }, path)
        return path
    
    def load_model(self, filename):
        """Load model from checkpoint."""
        path = os.path.join(self.model_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_log = checkpoint['training_log']
        return checkpoint['epoch']
    
    def save_training_curve(self):
        """Save the training loss curve as a PDF."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_log['epoch'], self.training_log['loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('ADNLCC Training Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        filename = os.path.join(self.logs_dir, 'training_loss_curve.pdf')
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        return filename

def train_model(model, train_loader, device, config, use_adnlcc=True, num_epochs=10):
    """Main training function."""
    trainer = ADNLCCTrainer(model, device, config)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for epoch in range(num_epochs):
        avg_loss, avg_correction = trainer.train_epoch(
            train_loader, epoch, noise_levels, use_adnlcc=use_adnlcc
        )
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Correction: {avg_correction:.6f}")
        
        if (epoch + 1) % 5 == 0:
            trainer.save_model(epoch + 1)
    
    trainer.save_model(num_epochs, "adnlcc_model_final.pt")
    curve_path = trainer.save_training_curve()
    print(f"Training curve saved to: {curve_path}")
    
    return trainer
