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
    
    # Process in smaller batches to reduce memory usage
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [SID]")
    for batch_idx, (data, _) in enumerate(pbar):
        # Print memory usage every 10 batches
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            print(f"  Batch {batch_idx}: GPU Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
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
        
        # Clear intermediate variables to free memory
        del gen, score_true, score_fake
        torch.cuda.empty_cache()
        
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
    for batch_idx, (data, _) in enumerate(pbar):
        # Print memory usage every 10 batches
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            print(f"  Batch {batch_idx}: GPU Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
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
        
        # Clear intermediate variables to free memory
        del gen_clean, gen_noisy, latent_clean, latent_noisy, data_noisy
        torch.cuda.empty_cache()
        
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
    
    # Set PyTorch CUDA memory allocation configuration to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Get device
    device = get_device(config['experiment']['gpu_id'])
    print(f"Using device: {device}")
    
    # Print GPU memory information if available
    if torch.cuda.is_available():
        print(f"GPU Memory before model creation:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"  Free: {torch.cuda.get_device_properties(device).total_memory / 1024**3 - torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    # Set up model - create on CPU first
    if config['model']['architecture'] == 'simple_cnn':
        # Create TwiST-Distill model
        model = TwiSTModel(
            in_channels=config['model']['in_channels'],
            feature_dim=config['model']['feature_dim'],
            hidden_dim=config['model']['hidden_dim'],
            use_bn=config['model']['use_batch_norm']
        )
        
        # For comparison, create SID model too (but keep on CPU initially)
        sid_model = SIDModel(
            in_channels=config['model']['in_channels'],
            feature_dim=config['model']['feature_dim'],
            hidden_dim=config['model']['hidden_dim'],
            use_bn=config['model']['use_batch_norm']
        )
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
        
    # Keep model on CPU initially
    # We'll move it to GPU in smaller chunks during training
    
    # Print GPU memory before any model is moved to device
    if torch.cuda.is_available():
        print(f"GPU Memory before loading any models:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
    
    twist_losses = []
    sid_losses = []
    consistency_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Move TwiST model to device for training
        print(f"Moving TwiST model to device for epoch {epoch}...")
        model = model.to(device)
        
        # Print GPU memory after moving TwiST model to device
        if torch.cuda.is_available():
            print(f"GPU Memory after moving TwiST model to device:")
            print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
        
        # Train TwiST-Distill model
        twist_loss, consistency_loss = train_twist(model, train_loader, optimizer, device, epoch, config)
        twist_losses.append(twist_loss)
        consistency_losses.append(consistency_loss)
        
        # Move TwiST model back to CPU and clear cache before switching models
        print(f"Moving TwiST model back to CPU...")
        model = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory after TwiST training (model moved to CPU):")
            print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
        
        # Move SID model to device and train
        print(f"Moving SID model to device for epoch {epoch}...")
        sid_model = sid_model.to(device)
        
        # Print GPU memory after moving SID model to device
        if torch.cuda.is_available():
            print(f"GPU Memory after moving SID model to device:")
            print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
        
        # Train baseline SID model for comparison
        sid_loss = train_sid(sid_model, train_loader, sid_optimizer, device, epoch, config)
        sid_losses.append(sid_loss)
        
        # Move SID model back to CPU and clear cache after training
        print(f"Moving SID model back to CPU...")
        sid_model = sid_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory after SID training (model moved to CPU):")
            print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
            print(f"GPU Memory after SID training:")
            print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
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
            
            # Generate and save sample images (on CPU to save memory)
            print(f"Generating sample images for epoch {epoch}...")
            
            # Process models one at a time to minimize memory usage
            with torch.no_grad():
                # First, get a small batch of test data
                data_sample, _ = next(iter(test_loader))
                # Use only 2 samples to further reduce memory usage (was 4)
                data_sample = data_sample[:2].to(device)
                
                # Print memory usage before image generation
                if torch.cuda.is_available():
                    print(f"GPU Memory before image generation:")
                    print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
                    print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
                
                # Create noisy version for testing
                data_noisy = add_controlled_noise(data_sample, config['training']['noise_std'])
                
                # Process TwiST model first
                print("Processing TwiST model images...")
                model.eval()
                model = model.to(device)
                gen_twist, _ = model(data_sample)
                gen_twist_noisy, _ = model(data_noisy)
                # Move model back to CPU immediately
                model = model.cpu()
                torch.cuda.empty_cache()
                
                # Process SID model next
                print("Processing SID model images...")
                sid_model.eval()
                sid_model = sid_model.to(device)
                gen_sid, _, _ = sid_model(data_sample)
                gen_sid_noisy, _, _ = sid_model(data_noisy)
                # Move model back to CPU immediately
                sid_model = sid_model.cpu()
                torch.cuda.empty_cache()
                
                # Print memory usage after image generation
                if torch.cuda.is_available():
                    print(f"GPU Memory after image generation:")
                    print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
                    print(f"  Free: {(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3:.2f} GB")
                
                # Concatenate for visual comparison
                print("Creating comparison image...")
                comparison = torch.cat([
                    data_sample, data_noisy,
                    gen_twist, gen_twist_noisy,
                    gen_sid, gen_sid_noisy
                ], dim=0)
                
                # Save comparison image
                save_path = os.path.join(config['training']['save_dir'], f"comparison_epoch{epoch}.png")
                save_images(comparison, save_path, nrow=2)  # Further reduced from 4 to 2
                print(f"Sample images saved to {save_path}")
    
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
