import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import argparse

from utils.utils import (
    load_config, set_seed, get_device, get_optimizer,
    get_scheduler, save_model, save_images
)
from utils.networks import SIDModel, TwiSTModel
from utils.metrics import compute_psnr, compute_ssim, add_controlled_noise

def train_sid(model, train_loader, optimizer, device, epoch, config):
    """Train SID model for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [SID]")
    for data, _ in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        gen, score_true, score_fake = model(data)
        
        # SID loss: MSE between true and fake scores, plus reconstruction loss
        score_loss = F.mse_loss(score_fake, score_true)
        recon_loss = F.mse_loss(gen, data)
        loss = score_loss + recon_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def train_twist(model, train_loader, optimizer, device, epoch, config):
    """Train TwiST-Distill model for one epoch."""
    model.train()
    total_loss = 0
    total_consistency_loss = 0
    noise_std = config['training']['noise_std']
    consistency_weight = config['training']['consistency_weight']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TwiST]")
    for data, _ in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass on clean data
        gen_clean, latent_clean = model(data)
        
        # Forward pass on noisy data
        data_noisy = add_controlled_noise(data, noise_std)
        gen_noisy, latent_noisy = model(data_noisy)
        
        # Compute losses
        # Reconstruction loss
        recon_loss_clean = F.mse_loss(gen_clean, data)
        recon_loss_noisy = F.mse_loss(gen_noisy, data)
        
        # Double-Tweedie consistency loss
        consistency_loss = F.mse_loss(latent_clean, latent_noisy)
        
        # Combined loss
        loss = recon_loss_clean + recon_loss_noisy + consistency_weight * consistency_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_consistency_loss += consistency_loss.item()
        pbar.set_postfix({"loss": loss.item(), "cons_loss": consistency_loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    avg_consistency_loss = total_consistency_loss / len(train_loader)
    
    return avg_loss, avg_consistency_loss

def train_model(config_path):
    """
    Train model based on configuration.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Trained model
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Get device
    device = get_device(config['experiment']['gpu_id'])
    print(f"Using device: {device}")
    
    # Set up model
    if config['model']['architecture'] == 'simple_cnn':
        # Create TwiST-Distill model
        model = TwiSTModel(
            in_channels=config['model']['in_channels'],
            feature_dim=config['model']['feature_dim'],
            hidden_dim=config['model']['hidden_dim'],
            use_bn=config['model']['use_batch_norm']
        ).to(device)
        
        # For comparison, create SID model too
        sid_model = SIDModel(
            in_channels=config['model']['in_channels'],
            feature_dim=config['model']['feature_dim'],
            hidden_dim=config['model']['hidden_dim'],
            use_bn=config['model']['use_batch_norm']
        ).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {config['model']['architecture']}")
    
    # Get data loaders
    from preprocess import preprocess_data
    train_loader, test_loader = preprocess_data(config_path)
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(model, config)
    sid_optimizer = get_optimizer(sid_model, config)
    scheduler = get_scheduler(optimizer, config, config['training']['num_epochs'])
    sid_scheduler = get_scheduler(sid_optimizer, config, config['training']['num_epochs'])
    
    # Create directory for saving models
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    if config['experiment']['test_run']:
        print("Running in test mode with reduced epochs")
        num_epochs = min(2, num_epochs)  # Limit to 2 epochs for testing
    
    twist_losses = []
    sid_losses = []
    consistency_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train TwiST-Distill model
        twist_loss, consistency_loss = train_twist(model, train_loader, optimizer, device, epoch, config)
        twist_losses.append(twist_loss)
        consistency_losses.append(consistency_loss)
        
        # Train baseline SID model for comparison
        sid_loss = train_sid(sid_model, train_loader, sid_optimizer, device, epoch, config)
        sid_losses.append(sid_loss)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        if sid_scheduler:
            sid_scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  TwiST Loss: {twist_loss:.6f}, Consistency Loss: {consistency_loss:.6f}")
        print(f"  SID Loss: {sid_loss:.6f}")
        
        # Save model if needed
        if epoch % config['training']['save_frequency'] == 0 or epoch == num_epochs:
            twist_save_path = os.path.join(config['training']['save_dir'], f"twist_model_epoch{epoch}.pth")
            sid_save_path = os.path.join(config['training']['save_dir'], f"sid_model_epoch{epoch}.pth")
            save_model(model, epoch, twist_save_path)
            save_model(sid_model, epoch, sid_save_path)
            
            # Generate and save sample images
            model.eval()
            sid_model.eval()
            with torch.no_grad():
                data_sample, _ = next(iter(test_loader))
                data_sample = data_sample[:8].to(device)  # Use 8 samples
                
                # Generate images with both models
                gen_twist, _ = model(data_sample)
                gen_sid, _, _ = sid_model(data_sample)
                
                # Add noise to test robustness
                data_noisy = add_controlled_noise(data_sample, config['training']['noise_std'])
                gen_twist_noisy, _ = model(data_noisy)
                gen_sid_noisy, _, _ = sid_model(data_noisy)
                
                # Concatenate for visual comparison
                comparison = torch.cat([
                    data_sample, data_noisy,
                    gen_twist, gen_twist_noisy,
                    gen_sid, gen_sid_noisy
                ], dim=0)
                
                # Save comparison image
                save_path = os.path.join(config['training']['save_dir'], f"comparison_epoch{epoch}.png")
                save_images(comparison, save_path, nrow=8)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Save final models
    final_twist_path = os.path.join(config['training']['save_dir'], "twist_model_final.pth")
    final_sid_path = os.path.join(config['training']['save_dir'], "sid_model_final.pth")
    save_model(model, num_epochs, final_twist_path)
    save_model(sid_model, num_epochs, final_sid_path)
    
    return model, sid_model, train_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TwiST-Distill model")
    parser.add_argument("--config", type=str, default="config/twist_distill_config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()
    
    model, sid_model, train_loader, test_loader = train_model(args.config)
    print("Training completed successfully.")
