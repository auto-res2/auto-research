"""
Training module for the Latent-Integrated Fingerprint Diffusion (LIFD) method.

This module provides functionality for:
1. Training the LIFD model with dual-channel fingerprinting
2. Implementing parameter-level modulation and latent-space conditioning
3. Implementing the adaptive balancing mechanism
4. Training the fingerprint extraction network
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

from preprocess import set_seed, simulate_attacks


class CustomCrossAttention(nn.Module):
    """
    A cross-attention module for latent fingerprint injection.
    It also saves an attention map during a forward pass.
    """
    def __init__(self, in_channels=3, fingerprint_dim=128):
        super(CustomCrossAttention, self).__init__()
        self.fingerprint_proj = nn.Linear(fingerprint_dim, 64)
        
        self.query = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.key = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.attention_map = None

    def forward(self, x, fingerprint=None):
        """
        Forward pass of the cross-attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            fingerprint (torch.Tensor): Fingerprint tensor of shape [B, fingerprint_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W]
        """
        batch_size, channels, height, width = x.shape
        
        if fingerprint is None:
            fingerprint = torch.zeros(batch_size, self.fingerprint_proj.in_features, device=x.device)
        
        fp_proj = self.fingerprint_proj(fingerprint)  # [B, 64]
        
        q = self.query(x)  # [B, 64, H, W]
        k = self.key(x)    # [B, 64, H, W]
        v = self.value(x)  # [B, C, H, W]
        
        q = q.view(batch_size, 64, -1)  # [B, 64, H*W]
        k = k.view(batch_size, 64, -1)  # [B, 64, H*W]
        v = v.view(batch_size, channels, -1)  # [B, C, H*W]
        
        attn = torch.bmm(q.permute(0, 2, 1), k)  # [B, H*W, H*W]
        attn = F.softmax(attn / np.sqrt(64), dim=2)
        
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(batch_size, channels, height, width)  # [B, C, H, W]
        
        fp_mod = fp_proj.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        attn_map = torch.sigmoid(torch.sum(q.view(batch_size, 64, height, width) * fp_mod, dim=1, keepdim=True))  # [B, 1, H, W]
        
        self.attention_map = attn_map
        
        out = out * attn_map + x
        
        out = self.out_proj(out)
        
        return out

def total_variation(x):
    """
    Calculate the total variation of an attention map tensor.
    
    Args:
        x (torch.Tensor): Tensor of shape [N, C, H, W]
        
    Returns:
        torch.Tensor: Total variation loss
    """
    tv_h = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return (tv_h + tv_w) / x.numel()


class LIFDModel(nn.Module):
    """
    Latent-Integrated Fingerprint Diffusion (LIFD) model.
    
    This model implements the dual-channel fingerprinting approach:
    1. Parameter-level modulation: Embeds fingerprints via weight modulation
    2. Latent-space conditioning: Injects fingerprints via cross-attention
    """
    def __init__(self, fingerprint_dim=128, image_size=64, adaptive_alpha=0.5):
        super(LIFDModel, self).__init__()
        self.fingerprint_dim = fingerprint_dim
        self.image_size = image_size
        self.adaptive_alpha = adaptive_alpha
        self.mode = 'C'  # Default to dual-channel mode
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 16x16
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        hidden_dim1 = max(fingerprint_dim * 2, 32)
        hidden_dim2 = max(hidden_dim1 * 2, 64)
        
        self.fingerprint_mapper = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 128),  # Output dimension matches the bottleneck dimension
        )
        
        self.cross_attention = CustomCrossAttention(in_channels=128, fingerprint_dim=fingerprint_dim)
    
    def set_mode(self, mode):
        """
        Set the mode: 'A' (Parameter-Only), 'B' (Latent-Only), or 'C' (Dual-Channel).
        
        Args:
            mode (str): Mode to set
        """
        assert mode in ['A', 'B', 'C'], "Mode must be 'A', 'B', or 'C'"
        self.mode = mode
        print(f"Model set to mode: {mode}")
    
    def set_adaptive_balance(self, alpha):
        """
        Set the adaptive balancing parameter (0 ≤ alpha ≤ 1).
        
        Args:
            alpha (float): Adaptive balancing parameter
        """
        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
        self.adaptive_alpha = alpha
        print(f"Adaptive balance parameter set to: {alpha}")
    
    def forward(self, x, fingerprint=None):
        """
        Forward pass of the LIFD model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W]
            fingerprint (torch.Tensor): Fingerprint tensor of shape [B, fingerprint_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, 3, H, W]
        """
        features = self.encoder(x)
        
        if self.mode == 'A':  # Parameter-Only
            if fingerprint is not None:
                fp_mod = self.fingerprint_mapper(fingerprint)
                fp_mod = fp_mod.unsqueeze(-1).unsqueeze(-1)  # [B, 128, 1, 1]
                features = features + fp_mod
        
        elif self.mode == 'B':  # Latent-Only
            if fingerprint is not None:
                features = self.cross_attention(features, fingerprint)
        
        elif self.mode == 'C':  # Dual-Channel
            if fingerprint is not None:
                fp_mod = self.fingerprint_mapper(fingerprint)
                fp_mod = fp_mod.unsqueeze(-1).unsqueeze(-1)  # [B, 128, 1, 1]
                features = features + (1 - self.adaptive_alpha) * fp_mod
                
                features = self.cross_attention(features, fingerprint)
        
        output = self.decoder(features)
        
        return output


class FingerprintExtractionNet(nn.Module):
    """
    Network for extracting fingerprints from generated images.
    """
    def __init__(self, fingerprint_dim=128, image_size=64):
        super(FingerprintExtractionNet, self).__init__()
        self.fingerprint_dim = fingerprint_dim
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  # 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  # 16x16
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)  # 8x8
        self.bn4 = nn.BatchNorm2d(256)
        
        hidden_dim = max(fingerprint_dim * 4, 64)
        
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fingerprint_dim)
    
    def forward(self, x):
        """
        Forward pass of the fingerprint extraction network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W]
            
        Returns:
            torch.Tensor: Extracted fingerprint of shape [B, fingerprint_dim]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def train_lifd_model(train_data, val_data, fingerprints, config=None):
    """
    Train the LIFD model.
    
    Args:
        train_data (list): List of training data batches
        val_data (list): List of validation data batches
        fingerprints (torch.Tensor): Tensor of fingerprints
        config (dict): Configuration parameters
        
    Returns:
        tuple: Trained LIFD model and extraction network
    """
    if config is None:
        config = {
            'seed': 42,
            'batch_size': 16,
            'image_size': 64,
            'fingerprint_dim': 128,
            'num_users': 10,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'adaptive_alpha': 0.5,
            'mode': 'C',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_dir': 'models',
            'log_dir': 'logs'
        }
    
    set_seed(config['seed'])
    
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    lifd_model = LIFDModel(
        fingerprint_dim=config['fingerprint_dim'],
        image_size=config['image_size'],
        adaptive_alpha=config['adaptive_alpha']
    ).to(device)
    
    extraction_net = FingerprintExtractionNet(
        fingerprint_dim=config['fingerprint_dim'],
        image_size=config['image_size']
    ).to(device)
    
    lifd_model.set_mode(config['mode'])
    
    lifd_optimizer = optim.Adam(
        lifd_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    extraction_optimizer = optim.Adam(
        extraction_net.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    print(f"Starting training for {config['num_epochs']} epochs")
    
    train_losses = []
    val_losses = []
    extraction_accuracies = []
    
    for epoch in range(config['num_epochs']):
        lifd_model.train()
        extraction_net.train()
        
        epoch_loss = 0.0
        epoch_extraction_acc = 0.0
        
        for batch_idx, batch in enumerate(train_data):
            batch = batch.to(device)
            
            batch_size = batch.size(0)
            user_indices = torch.randint(0, config['num_users'], (batch_size,))
            batch_fingerprints = fingerprints[user_indices].to(device)
            
            generated_images = lifd_model(batch, batch_fingerprints)
            
            recon_loss = mse_loss(generated_images, batch)
            
            tv_loss = 0.0
            if lifd_model.mode in ['B', 'C'] and lifd_model.cross_attention.attention_map is not None:
                tv_loss = total_variation(lifd_model.cross_attention.attention_map)
            
            extracted_fingerprints = extraction_net(generated_images)
            
            extraction_loss = bce_loss(extracted_fingerprints, batch_fingerprints)
            
            total_loss = recon_loss + 0.1 * tv_loss + extraction_loss
            
            lifd_optimizer.zero_grad()
            extraction_optimizer.zero_grad()
            total_loss.backward()
            lifd_optimizer.step()
            extraction_optimizer.step()
            
            with torch.no_grad():
                pred_fingerprints = (torch.sigmoid(extracted_fingerprints) > 0.5).float()
                accuracy = (pred_fingerprints == batch_fingerprints).float().mean().item()
            
            epoch_loss += total_loss.item()
            epoch_extraction_acc += accuracy
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{config['num_epochs']}, Batch {batch_idx+1}/{len(train_data)}, "
                      f"Loss: {total_loss.item():.4f}, Extraction Acc: {accuracy:.4f}")
        
        epoch_loss /= len(train_data)
        epoch_extraction_acc /= len(train_data)
        
        lifd_model.eval()
        extraction_net.eval()
        
        val_loss = 0.0
        val_extraction_acc = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                batch = batch.to(device)
                
                batch_size = batch.size(0)
                user_indices = torch.randint(0, config['num_users'], (batch_size,))
                batch_fingerprints = fingerprints[user_indices].to(device)
                
                generated_images = lifd_model(batch, batch_fingerprints)
                
                recon_loss = mse_loss(generated_images, batch)
                
                extracted_fingerprints = extraction_net(generated_images)
                
                extraction_loss = bce_loss(extracted_fingerprints, batch_fingerprints)
                
                total_loss = recon_loss + extraction_loss
                
                pred_fingerprints = (torch.sigmoid(extracted_fingerprints) > 0.5).float()
                accuracy = (pred_fingerprints == batch_fingerprints).float().mean().item()
                
                val_loss += total_loss.item()
                val_extraction_acc += accuracy
        
        val_loss /= len(val_data)
        val_extraction_acc /= len(val_data)
        
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        extraction_accuracies.append(val_extraction_acc)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_extraction_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_extraction_acc:.4f}")
        
        if (epoch + 1) % 5 == 0 or epoch == config['num_epochs'] - 1:
            torch.save(lifd_model.state_dict(), os.path.join(config['save_dir'], f"lifd_model_epoch{epoch+1}.pth"))
            torch.save(extraction_net.state_dict(), os.path.join(config['save_dir'], f"extraction_net_epoch{epoch+1}.pth"))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(extraction_accuracies, label='Extraction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fingerprint Extraction Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['log_dir'], 'training_curves.pdf'), format='pdf', dpi=300)
    print(f"Training curves saved to {os.path.join(config['log_dir'], 'training_curves.pdf')}")
    
    return lifd_model, extraction_net

def train_model(preprocessed_data, config=None):
    """
    Main training function that trains the LIFD model.
    
    Args:
        preprocessed_data (dict): Preprocessed data and metadata
        config (dict): Configuration parameters
        
    Returns:
        dict: Trained models and training metrics
    """
    if config is None:
        config = {
            'seed': 42,
            'batch_size': 16,
            'image_size': 64,
            'fingerprint_dim': 128,
            'num_users': 10,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'adaptive_alpha': 0.5,
            'mode': 'C',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_dir': 'models',
            'log_dir': 'logs'
        }
    
    train_data = preprocessed_data['train_data']
    val_data = preprocessed_data['val_data']
    fingerprints = preprocessed_data['fingerprints']
    
    lifd_model, extraction_net = train_lifd_model(
        train_data=train_data,
        val_data=val_data,
        fingerprints=fingerprints,
        config=config
    )
    
    trained_models = {
        'lifd_model': lifd_model,
        'extraction_net': extraction_net,
        'config': config
    }
    
    print("Model training completed successfully")
    return trained_models

if __name__ == "__main__":
    from preprocess import preprocess_data
    
    config = {
        'seed': 42,
        'batch_size': 16,
        'image_size': 64,
        'use_dummy': True,
        'data_dir': None,
        'fingerprint_dim': 128,
        'num_users': 10,
        'num_epochs': 2,  # Small number for testing
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'adaptive_alpha': 0.5,
        'mode': 'C',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'models',
        'log_dir': 'logs'
    }
    
    preprocessed_data = preprocess_data(config)
    
    trained_models = train_model(preprocessed_data, config)
    
    print("Training test completed successfully")
