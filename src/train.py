#!/usr/bin/env python3
"""
Model training module for CGCD experiments.

This module implements:
1. Base Diffusion Model
2. CGCD Diffusion Model with Adaptive Guidance
3. Model variants for ablation studies
4. Continuous CGCD model
5. Training loops for all experiment types
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.diffusion_utils import (
    seed_everything, get_device, save_model, 
    create_tensorboard_writer, plot_images, get_condition
)

# Base Diffusion Model with hard conditioning
class BaseDiffusionModel(nn.Module):
    def __init__(self):
        super(BaseDiffusionModel, self).__init__()
        # Define a simple encoder-decoder network for demonstration
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, condition, noise_level, alpha_t):
        # Run encoder
        z = self.encoder(x)
        # Hard conditioning: simply add condition (expanded to spatial dims)
        # Get dimensions
        batch_size = z.shape[0]
        feature_dim = z.shape[1]  # Number of feature channels
        spatial_size = z.shape[2:]
        
        # Ensure condition has the right shape for broadcasting
        # First ensure condition has the right number of channels (feature_dim)
        if condition.shape[1] != feature_dim:
            # Create a projection layer on the fly if dimensions don't match
            projection = nn.Linear(condition.shape[1], feature_dim).to(x.device)
            condition = projection(condition)
        
        # Reshape and expand to match spatial dimensions
        condition_expanded = condition.view(batch_size, feature_dim, 1, 1).expand(-1, -1, *spatial_size)
        z = z + condition_expanded
        
        # Reverse diffusion (dummy update with fixed noise schedule)
        z = z * (1 - noise_level) + noise_level * torch.randn_like(z)
        # Multiply by static alpha_t
        z = z * alpha_t  
        # Decoder output
        out = self.decoder(z)
        return out

# CGCD Diffusion Model with Adaptive Guidance
class CGCDDiffusionModel(BaseDiffusionModel):
    def __init__(self):
        super().__init__()
        # Additional module for adaptive guidance: soft target assimilation
        self.adaptive_module = nn.Linear(10, 64)  # assume condition has 10 dimensions
    
    def forward(self, x, condition, noise_level, alpha_t):
        z = self.encoder(x)
        # Soft target assimilation: use adaptive module on condition
        condition_embedded = self.adaptive_module(condition.float())
        z = z + condition_embedded.unsqueeze(-1).unsqueeze(-1)
        # Adaptive noise schedule: update noise using an adaptive function
        dynamic_noise = noise_level * (torch.sigmoid(torch.mean(z)) + 0.5)
        z = z * (1 - dynamic_noise) + dynamic_noise * torch.randn_like(z)
        # Update dynamic alpha: Here we use a function based on z's mean
        dynamic_alpha = alpha_t * (1 + 0.1 * torch.sigmoid(torch.mean(z)))
        z = z * dynamic_alpha
        out = self.decoder(z)
        return out, dynamic_alpha

# Diffusion Model for Ablation Studies
class DiffusionModelVariant(BaseDiffusionModel):
    def __init__(self, adaptive_noise=True, soft_assimilation=True):
        super(DiffusionModelVariant, self).__init__()
        self.adaptive_noise = adaptive_noise
        self.soft_assimilation = soft_assimilation
        if self.soft_assimilation:
            # Mapping condition to latent space dimension (assume condition is 10-d)
            self.assimilation_module = nn.Linear(10, 64)
    
    def forward(self, x, condition, noise_level, alpha_t):
        z = self.encoder(x)
        # Conditioning: use soft assimilation if enabled; otherwise, hard conditioning
        if self.soft_assimilation:
            # Get dimensions for proper reshaping
            batch_size = z.shape[0]
            feature_dim = z.shape[1]
            spatial_size = z.shape[2:]
            
            cond_emb = self.assimilation_module(condition.float())
            # Reshape and expand to match spatial dimensions
            cond_emb_expanded = cond_emb.view(batch_size, feature_dim, 1, 1).expand(-1, -1, *spatial_size)
            z = z + cond_emb_expanded
        else:
            # Get dimensions for proper reshaping
            batch_size = z.shape[0]
            feature_dim = z.shape[1]
            spatial_size = z.shape[2:]
            
            # Ensure condition has the right shape for broadcasting
            if condition.shape[1] != feature_dim:
                # Create a projection layer on the fly if dimensions don't match
                projection = nn.Linear(condition.shape[1], feature_dim).to(x.device)
                condition = projection(condition)
            
            # Reshape and expand to match spatial dimensions
            condition_expanded = condition.view(batch_size, feature_dim, 1, 1).expand(-1, -1, *spatial_size)
            z = z + condition_expanded
        # Noise update
        if self.adaptive_noise:
            dynamic_noise = noise_level * (torch.sigmoid(torch.mean(z)) + 0.5)
        else:
            dynamic_noise = noise_level
        z = z * (1 - dynamic_noise) + dynamic_noise * torch.randn_like(z)
        # Alpha update if adaptive noise is on, else static
        if self.adaptive_noise:
            dynamic_alpha = alpha_t * (1 + 0.1 * torch.sigmoid(torch.mean(z)))
        else:
            dynamic_alpha = alpha_t
        z = z * dynamic_alpha
        out = self.decoder(z)
        return out, dynamic_alpha

# Model for Continuous Conditioning
class ContinuousCGCDModel(BaseDiffusionModel):
    def __init__(self):
        super(ContinuousCGCDModel, self).__init__()
        # MLP that maps a continuous scalar signal to the latent space
        self.continuous_mapper = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

    def forward(self, x, cont_signal, noise_level, alpha_t):
        z = self.encoder(x)
        # Map continuous signal into latent space and add
        # Get dimensions
        batch_size = z.shape[0]
        feature_dim = z.shape[1]
        spatial_size = z.shape[2:]
        
        signal_emb = self.continuous_mapper(cont_signal.view(-1, 1))
        
        # Ensure signal embedding has the right feature dimension
        if signal_emb.shape[1] != feature_dim:
            # Create a projection layer on the fly if dimensions don't match
            projection = nn.Linear(signal_emb.shape[1], feature_dim).to(x.device)
            signal_emb = projection(signal_emb)
            
        # Reshape and expand to match spatial dimensions
        signal_emb_expanded = signal_emb.view(batch_size, feature_dim, 1, 1).expand(-1, -1, *spatial_size)
        z = z + signal_emb_expanded
        dynamic_noise = noise_level * (torch.sigmoid(torch.mean(z)) + 0.5)
        z = z * (1 - dynamic_noise) + dynamic_noise * torch.randn_like(z)
        dynamic_alpha = alpha_t * (1 + 0.1 * torch.sigmoid(torch.mean(z)))
        z = z * dynamic_alpha
        out = self.decoder(z)
        return out, dynamic_alpha

# Training loop for Experiment 1 for both Base and CGCD methods
def train_experiment1(model, optimizer, dataloader, model_name='Base', 
                     epochs=1, max_batches=None, device=None, 
                     log_dir='logs', model_dir='models',
                     test_mode=False):
    """
    Train models for Experiment 1: Performance Comparison.
    
    Args:
        model: Model to train (Base or CGCD)
        optimizer: Optimizer for training
        dataloader: DataLoader for training data
        model_name: Name of the model ('Base' or 'CGCD')
        epochs: Number of training epochs
        max_batches: Maximum number of batches per epoch (for testing)
        device: Device to train on
        log_dir: Directory for tensorboard logs
        model_dir: Directory to save models
        test_mode: Whether to run in test mode with minimal computations
    """
    if device is None:
        device = get_device()
        
    model.to(device)
    model.train()
    
    # Create log directories
    writer = create_tensorboard_writer(f'{log_dir}/{model_name}_Exp1')
    
    print(f"\n{'='*50}")
    print(f"Starting training for {model_name} (Experiment 1)")
    print(f"{'='*50}")
    
    global_step = 0
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches_processed = 0
        
        for i, (data, labels) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
                
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            condition = get_condition(labels).to(device)
            noise_level = 0.2  # fixed noise level
            alpha_t = torch.tensor(1.0).to(device)
            
            if model_name == 'CGCD':
                output, dynamic_alpha = model(data, condition, noise_level, alpha_t)
                writer.add_scalar("DynamicAlpha", dynamic_alpha.mean().item(), global_step)
            else:
                output = model(data, condition, noise_level, alpha_t)
            
            loss = loss_fn(output, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches_processed += 1
            
            if i % 5 == 0 or test_mode:
                print(f"[Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")
                writer.add_scalar("Loss", loss.item(), global_step)
                
                # In test mode, also save a sample output
                if test_mode and i == 0:
                    if model_name == 'CGCD':
                        sample_output = output
                    else:
                        sample_output = output
                    plot_images(
                        sample_output[:16], 
                        f"{model_name} Output - Epoch {epoch+1}",
                        save_path=f"{log_dir}/{model_name}_sample_e{epoch+1}.png"
                    )
            
            global_step += 1
        
        # Epoch summary
        if batches_processed > 0:
            avg_loss = epoch_loss / batches_processed
            print(f"Epoch {epoch+1}/{epochs} Summary - Avg Loss: {avg_loss:.4f}")
            writer.add_scalar("Epoch_Loss", avg_loss, epoch)
        
        # Save model checkpoint
        if not test_mode:
            save_model(model, model_dir, f"{model_name}_epoch{epoch+1}.pt")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"{'='*50}\n")
    
    writer.close()
    return model

# Function to train a variant (Experiment 2) for ablation studies
def train_experiment2(model, optimizer, dataloader, variant_name='Variant', 
                     epochs=1, max_batches=None, device=None,
                     log_dir='logs', model_dir='models',
                     test_mode=False):
    """
    Train models for Experiment 2: Ablation Study.
    
    Args:
        model: Model variant to train
        optimizer: Optimizer for training
        dataloader: DataLoader for training data
        variant_name: Name of the model variant
        epochs: Number of training epochs
        max_batches: Maximum number of batches per epoch (for testing)
        device: Device to train on
        log_dir: Directory for tensorboard logs
        model_dir: Directory to save models
        test_mode: Whether to run in test mode with minimal computations
    """
    if device is None:
        device = get_device()
        
    model.to(device)
    model.train()
    
    # Create log directories
    writer = create_tensorboard_writer(f'{log_dir}/{variant_name}_Exp2')
    
    print(f"\n{'='*50}")
    print(f"Starting ablation training for {variant_name} (Experiment 2)")
    print(f"{'='*50}")
    
    global_step = 0
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches_processed = 0
        
        for i, (data, labels) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
                
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            condition = get_condition(labels).to(device)
            noise_level = 0.2
            alpha_t = torch.tensor(1.0).to(device)
            
            output, dynamic_alpha = model(data, condition, noise_level, alpha_t)
            
            loss = loss_fn(output, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches_processed += 1
            
            if i % 5 == 0 or test_mode:
                print(f"[{variant_name} | Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | Alpha: {dynamic_alpha.mean().item():.4f}")
                writer.add_scalar("Loss", loss.item(), global_step)
                writer.add_scalar("DynamicAlpha", dynamic_alpha.mean().item(), global_step)
                
                # In test mode, also save a sample output
                if test_mode and i == 0:
                    plot_images(
                        output[:16], 
                        f"{variant_name} Output - Epoch {epoch+1}",
                        save_path=f"{log_dir}/{variant_name}_sample_e{epoch+1}.png"
                    )
            
            global_step += 1
        
        # Epoch summary
        if batches_processed > 0:
            avg_loss = epoch_loss / batches_processed
            print(f"Epoch {epoch+1}/{epochs} Summary - Avg Loss: {avg_loss:.4f}")
            writer.add_scalar("Epoch_Loss", avg_loss, epoch)
        
        # Save model checkpoint
        if not test_mode:
            save_model(model, model_dir, f"{variant_name}_epoch{epoch+1}.pt")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"{'='*50}\n")
    
    writer.close()
    return model
    
# Training function for Experiment 3 (Continuous Features)
def train_experiment3(model, optimizer, dataloader, 
                     epochs=1, max_batches=None, device=None,
                     log_dir='logs', model_dir='models',
                     test_mode=False):
    """
    Train models for Experiment 3: Continuous Feature Domains.
    
    Args:
        model: Continuous CGCD model to train
        optimizer: Optimizer for training
        dataloader: DataLoader for synthetic continuous data
        epochs: Number of training epochs
        max_batches: Maximum number of batches per epoch (for testing)
        device: Device to train on
        log_dir: Directory for tensorboard logs
        model_dir: Directory to save models
        test_mode: Whether to run in test mode with minimal computations
    """
    if device is None:
        device = get_device()
        
    model.to(device)
    model.train()
    
    # Create log directories
    writer = create_tensorboard_writer(f'{log_dir}/ContinuousCGCD_Exp3')
    
    print(f"\n{'='*50}")
    print(f"Starting training for ContinuousCGCD Model (Experiment 3)")
    print(f"{'='*50}")
    
    global_step = 0
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches_processed = 0
        
        for i, (data, signal) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
                
            data = data.to(device)
            signal = signal.to(device)
            
            optimizer.zero_grad()
            noise_level = 0.2
            alpha_t = torch.tensor(1.0).to(device)
            
            output, dynamic_alpha = model(data, signal, noise_level, alpha_t)
            
            loss = loss_fn(output, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches_processed += 1
            
            if i % 5 == 0 or test_mode:
                print(f"[Continuous | Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | Alpha: {dynamic_alpha.mean().item():.4f}")
                writer.add_scalar("Loss", loss.item(), global_step)
                writer.add_scalar("DynamicAlpha", dynamic_alpha.mean().item(), global_step)
                
                # In test mode, also save a sample output
                if test_mode and i == 0:
                    plot_images(
                        output[:16], 
                        f"Continuous CGCD Output - Epoch {epoch+1}",
                        save_path=f"{log_dir}/Continuous_sample_e{epoch+1}.png"
                    )
            
            global_step += 1
        
        # Epoch summary
        if batches_processed > 0:
            avg_loss = epoch_loss / batches_processed
            print(f"Epoch {epoch+1}/{epochs} Summary - Avg Loss: {avg_loss:.4f}")
            writer.add_scalar("Epoch_Loss", avg_loss, epoch)
        
        # Save model checkpoint
        if not test_mode:
            save_model(model, model_dir, f"Continuous_epoch{epoch+1}.pt")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"{'='*50}\n")
    
    writer.close()
    
    # Post-training: perform a quick latent alignment analysis
    if not test_mode:
        analyze_continuous_latent_alignment(model, dataloader, device, log_dir)
    
    return model

def analyze_continuous_latent_alignment(model, dataloader, device, log_dir):
    """
    Analyze the alignment between continuous signals and latent features.
    
    Args:
        model: Trained continuous CGCD model
        dataloader: DataLoader for continuous data
        device: Device to run analysis on
        log_dir: Directory to save analysis plots
    """
    model.eval()
    all_signals = []
    all_latent_means = []
    
    with torch.no_grad():
        for data, signal in dataloader:
            data = data.to(device)
            signal = signal.to(device)
            
            # Get latent representation
            latent = model.encoder(data)
            latent_mean = latent.mean(dim=(1, 2, 3)).cpu().numpy()
            
            all_signals.extend(signal.cpu().numpy())
            all_latent_means.extend(latent_mean)
    
    # Create plot directory
    os.makedirs(f"{log_dir}/analysis", exist_ok=True)
    
    # Plot the relationship
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(all_signals, all_latent_means, alpha=0.5)
    plt.xlabel("Continuous Conditioning Signal")
    plt.ylabel("Average Latent Feature")
    plt.title("Latent Feature Alignment with Continuous Condition")
    plt.savefig(f"{log_dir}/analysis/latent_alignment.png")
    plt.close()
    
    print(f"Latent alignment analysis completed and saved to {log_dir}/analysis/latent_alignment.png")
