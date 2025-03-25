#!/usr/bin/env python
"""
Training module for Tweedie-Guided Global Consistent Video Editing (TG-GCVE).

This module implements:
1. TG-GCVE model architecture with dual-stage denoising
2. Tweedie-inspired consistency loss
3. Global context aggregation
4. Adaptive fusion and iterative refinement
5. Training loops for model variants
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
import lpips

# Import from preprocess.py
from preprocess import set_seed, preprocess_for_training, add_noise

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# Model Architecture
# --------------------------

class GlobalContextModule(nn.Module):
    """
    Global context aggregation module that computes attention maps
    across frames to enable global editing capabilities.
    """
    
    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Feature extraction
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Attention computation
        self.query_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Output projection
        self.output_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the global context module.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            tuple: (output tensor, attention map)
        """
        batch_size, _, height, width = x.shape
        
        # Extract features
        features = self.feature_conv(x)
        
        # Compute query, key, value
        query = self.query_conv(features).view(batch_size, self.hidden_dim, -1)
        key = self.key_conv(features).view(batch_size, self.hidden_dim, -1)
        value = self.value_conv(features).view(batch_size, self.hidden_dim, -1)
        
        # Transpose for attention computation
        query = query.permute(0, 2, 1)  # [B, H*W, C]
        key = key.permute(0, 2, 1)  # [B, H*W, C]
        value = value.permute(0, 2, 1)  # [B, H*W, C]
        
        # Compute attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [B, H*W, H*W]
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)  # Scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.bmm(attention_weights, value)  # [B, H*W, C]
        context = context.permute(0, 2, 1).contiguous()  # [B, C, H*W]
        context = context.view(batch_size, self.hidden_dim, height, width)
        
        # Project back to input dimension
        output = self.output_conv(context)
        
        # Extract attention map (average over query positions)
        attention_map = attention_weights.mean(dim=1).view(batch_size, height, width)
        
        return output, attention_map

class AdaptiveFusionModule(nn.Module):
    """
    Adaptive fusion module that combines outputs from the spatiotemporal
    denoiser and the global editing branch.
    """
    
    def __init__(self, in_channels=64):
        super().__init__()
        
        # Fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Attention gate for adaptive weighting
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x1, x2):
        """
        Forward pass through the adaptive fusion module.
        
        Args:
            x1 (torch.Tensor): First input tensor
            x2 (torch.Tensor): Second input tensor
            
        Returns:
            torch.Tensor: Fused output tensor
        """
        # Concatenate inputs
        concat = torch.cat([x1, x2], dim=1)
        
        # Compute attention weights
        weights = self.attention_gate(concat)
        
        # Apply weights to inputs
        weighted_x1 = x1 * weights[:, 0:1, :, :]
        weighted_x2 = x2 * weights[:, 1:2, :, :]
        
        # Concatenate weighted inputs
        weighted_concat = torch.cat([weighted_x1, weighted_x2], dim=1)
        
        # Apply fusion convolution
        output = self.fusion_conv(weighted_concat)
        
        return output

class TGGCVEModule(nn.Module):
    """
    Tweedie-Guided Global Consistent Video Editing (TG-GCVE) model.
    
    This model implements:
    1. Dual-stage denoising with Tweedie-inspired consistency
    2. Global context aggregation for global editing
    3. Adaptive fusion of local and global features
    4. Optional iterative refinement
    """
    
    def __init__(self, 
                 in_channels=3, 
                 hidden_dim=64, 
                 use_refined_stage=True, 
                 use_consistency_loss=True, 
                 use_global_context=True,
                 iterative_refinement=False,
                 num_iterations=3):
        super().__init__()
        
        print(f"Initializing TGGCVEModule: refined_stage={use_refined_stage}, "
              f"consistency_loss={use_consistency_loss}, global_context={use_global_context}, "
              f"iterative_refinement={iterative_refinement}, num_iterations={num_iterations}")
        
        self.use_refined_stage = use_refined_stage
        self.use_consistency_loss = use_consistency_loss
        self.use_global_context = use_global_context
        self.iterative_refinement = iterative_refinement
        self.num_iterations = num_iterations
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # First stage denoiser
        self.first_stage = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Second (refined) stage denoiser
        self.refined_stage = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ) if use_refined_stage else None
        
        # Global context module
        self.global_context = GlobalContextModule(hidden_dim) if use_global_context else None
        
        # Global editing branch
        self.global_edit = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ) if use_global_context else None
        
        # Adaptive fusion module
        self.fusion = AdaptiveFusionModule(hidden_dim) if use_global_context else None
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, x, noise_level=0.1, global_editing_intensity=1.0):
        """
        Forward pass through the TG-GCVE model.
        
        Args:
            x (torch.Tensor): Input tensor
            noise_level (float): Noise level for denoising
            global_editing_intensity (float): Intensity of global editing
            
        Returns:
            tuple: (output tensor, intermediate outputs)
        """
        # Encode input
        feat = self.encoder(x)
        
        # First stage denoising
        out_first = self.first_stage(feat)
        
        # Second stage (refined) denoising
        if self.refined_stage is not None:
            out_refined = self.refined_stage(out_first)
        else:
            out_refined = out_first
        
        # Global context and editing
        if self.global_context is not None and self.global_edit is not None and self.fusion is not None:
            global_feat, attn_map = self.global_context(feat)
            global_edit = self.global_edit(global_feat)
            
            # Apply global editing intensity
            global_edit = global_edit * global_editing_intensity
            
            # Fuse local and global features
            fused = self.fusion(out_refined, global_edit)
        else:
            fused = out_refined
            attn_map = None
        
        # Decode to output
        out = self.decoder(fused)
        
        # Iterative refinement
        if self.iterative_refinement:
            intermediate_outputs = [out]
            
            for i in range(self.num_iterations):
                # Re-encode output
                refined_feat = self.encoder(out)
                
                # Apply first stage denoising
                refined_first = self.first_stage(refined_feat)
                
                # Apply second stage denoising if available
                if self.refined_stage is not None:
                    refined_out = self.refined_stage(refined_first)
                else:
                    refined_out = refined_first
                
                # Apply global context and editing if available
                if self.global_context is not None and self.global_edit is not None and self.fusion is not None:
                    refined_global, _ = self.global_context(refined_feat)
                    refined_global_edit = self.global_edit(refined_global)
                    
                    # Apply global editing intensity
                    refined_global_edit = refined_global_edit * global_editing_intensity
                    
                    # Fuse local and global features
                    refined_fused = self.fusion(refined_out, refined_global_edit)
                else:
                    refined_fused = refined_out
                
                # Decode to output
                out = self.decoder(refined_fused)
                intermediate_outputs.append(out)
            
            return out, out_first, out_refined, attn_map, intermediate_outputs
        
        return out, out_first, out_refined, attn_map, None

# --------------------------
# Loss Functions
# --------------------------

def consistency_loss(out_first, out_refined):
    """
    Compute consistency loss between first and refined stage outputs.
    
    Args:
        out_first (torch.Tensor): Output from first denoising stage
        out_refined (torch.Tensor): Output from refined denoising stage
        
    Returns:
        torch.Tensor: Consistency loss
    """
    return torch.mean((out_first - out_refined) ** 2)

def compute_clip_score(img_tensor, text_description="edited video", clip_model=None):
    """
    Compute CLIP score between image and text description.
    
    Args:
        img_tensor (torch.Tensor): Image tensor
        text_description (str): Text description
        clip_model: CLIP model
        
    Returns:
        float: CLIP score
    """
    if clip_model is None:
        # Load CLIP model if not provided
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Prepare image and text inputs
    if img_tensor.shape[1] == 3:  # Check if image has 3 channels (RGB)
        # Normalize to [0, 1] range
        img_tensor = (img_tensor + 1) / 2  # Assuming input is in [-1, 1] range
        img_tensor = img_tensor.clamp(0, 1)
        
        # Resize to 224x224 if needed
        if img_tensor.shape[2] != 224 or img_tensor.shape[3] != 224:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=(224, 224), mode='bilinear', align_corners=False
            )
    else:
        raise ValueError("Image tensor must have 3 channels (RGB)")
    
    # Tokenize text
    text_input = clip.tokenize([text_description]).to(device)
    
    # Compute features
    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor)
        text_features = clip_model.encode_text(text_input)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        score = similarity.max().item()
    
    return score

# --------------------------
# Training Functions
# --------------------------

def train_epoch(model, dataloader, optimizer, lpips_loss_fn, epoch, config):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): DataLoader for training data
        optimizer (Optimizer): Optimizer for training
        lpips_loss_fn: LPIPS loss function
        epoch (int): Current epoch number
        config (dict): Training configuration
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Get frames from batch
        frames = batch['frames'].to(device)
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Reshape to [B*T, C, H, W] for processing individual frames
        frames = frames.view(-1, channels, height, width)
        
        # Add noise to frames
        noise_level = torch.tensor(config['noise_level']).to(device)
        noisy_frames = add_noise(frames, noise_level)
        
        # Forward pass
        optimizer.zero_grad()
        output, out_first, out_refined, attn_map, intermediates = model(
            noisy_frames, 
            noise_level=noise_level,
            global_editing_intensity=config['global_editing_intensity']
        )
        
        # Compute reconstruction loss
        rec_loss = nn.functional.mse_loss(output, frames)
        
        # Compute perceptual loss using LPIPS
        perc_loss = lpips_loss_fn(output, frames).mean()
        
        # Compute consistency loss if enabled
        if config['use_consistency_loss'] and model.refined_stage is not None:
            c_loss = consistency_loss(out_first, out_refined)
            loss = rec_loss + config['perceptual_weight'] * perc_loss + config['consistency_weight'] * c_loss
        else:
            loss = rec_loss + config['perceptual_weight'] * perc_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        epoch_loss += loss.item()
        progress_bar.set_postfix({
            'loss': loss.item(),
            'rec_loss': rec_loss.item(),
            'perc_loss': perc_loss.item()
        })
        
        # Print detailed loss every config['print_freq'] batches
        if batch_idx % config['print_freq'] == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}: "
                  f"Loss = {loss.item():.4f}, "
                  f"Rec Loss = {rec_loss.item():.4f}, "
                  f"Perc Loss = {perc_loss.item():.4f}")
            
            if config['use_consistency_loss'] and model.refined_stage is not None:
                print(f"Consistency Loss = {c_loss.item():.4f}")
    
    # Compute average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{config['epochs']}: Avg Loss = {avg_loss:.4f}")
    
    return avg_loss

def validate(model, dataloader, lpips_loss_fn, clip_model, config):
    """
    Validate the model on the validation set.
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): DataLoader for validation data
        lpips_loss_fn: LPIPS loss function
        clip_model: CLIP model
        config (dict): Training configuration
        
    Returns:
        dict: Validation metrics
    """
    model.eval()
    val_rec_loss = 0.0
    val_perc_loss = 0.0
    val_clip_score = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get frames from batch
            frames = batch['frames'].to(device)
            batch_size, seq_len, channels, height, width = frames.shape
            
            # Reshape to [B*T, C, H, W] for processing individual frames
            frames = frames.view(-1, channels, height, width)
            
            # Add noise to frames
            noise_level = torch.tensor(config['noise_level']).to(device)
            noisy_frames = add_noise(frames, noise_level)
            
            # Forward pass
            output, _, _, _, _ = model(
                noisy_frames, 
                noise_level=noise_level,
                global_editing_intensity=config['global_editing_intensity']
            )
            
            # Compute reconstruction loss
            rec_loss = nn.functional.mse_loss(output, frames)
            val_rec_loss += rec_loss.item()
            
            # Compute perceptual loss using LPIPS
            perc_loss = lpips_loss_fn(output, frames).mean()
            val_perc_loss += perc_loss.item()
            
            # Compute CLIP score for a sample image
            if batch_idx == 0:
                sample_output = output[0:1]  # Take first image
                clip_score = compute_clip_score(
                    sample_output, 
                    text_description=config['target_text'],
                    clip_model=clip_model
                )
                val_clip_score = clip_score
    
    # Compute average losses
    avg_rec_loss = val_rec_loss / len(dataloader)
    avg_perc_loss = val_perc_loss / len(dataloader)
    
    print(f"Validation: Rec Loss = {avg_rec_loss:.4f}, "
          f"Perc Loss = {avg_perc_loss:.4f}, "
          f"CLIP Score = {val_clip_score:.4f}")
    
    return {
        'rec_loss': avg_rec_loss,
        'perc_loss': avg_perc_loss,
        'clip_score': val_clip_score
    }

def train_variant(variant_config, dataloader, epochs=2):
    """
    Train a specific variant of the TG-GCVE model.
    
    Args:
        variant_config (dict): Configuration for the model variant
        dataloader (DataLoader): DataLoader for training data
        epochs (int): Number of epochs to train
        
    Returns:
        nn.Module: Trained model
    """
    print(f"\nTraining variant: {variant_config}")
    
    # Create model
    model = TGGCVEModule(
        in_channels=3,
        hidden_dim=64,
        use_refined_stage=variant_config["use_refined_stage"],
        use_consistency_loss=variant_config["use_consistency_loss"],
        use_global_context=variant_config["use_global_context"],
        iterative_refinement=variant_config.get("iterative_refinement", False),
        num_iterations=variant_config.get("num_iterations", 3)
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create LPIPS loss function
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            # Get frames from batch
            if isinstance(batch, dict):
                frames = batch['frames'].to(device)
            else:
                # Handle case where batch is a tuple (input, target)
                frames = batch[0].to(device)
            
            # Check if frames has 5 dimensions [B, T, C, H, W]
            if frames.dim() == 5:
                batch_size, seq_len, channels, height, width = frames.shape
                # Reshape to [B*T, C, H, W] for processing individual frames
                frames = frames.view(-1, channels, height, width)
            else:
                # Already in format [B, C, H, W]
                batch_size, channels, height, width = frames.shape
            
            # Add noise to frames
            noise_level = torch.tensor(0.1).to(device)
            noisy_frames = add_noise(frames, noise_level)
            
            # Forward pass
            optimizer.zero_grad()
            output, out_first, out_refined, _, _ = model(noisy_frames, noise_level)
            
            # Compute reconstruction loss
            rec_loss = nn.functional.mse_loss(output, frames)
            
            # Compute perceptual loss using LPIPS
            perc_loss = lpips_loss_fn(output, frames).mean()
            
            # Compute total loss
            loss = rec_loss + 0.1 * perc_loss
            
            # Add consistency loss if enabled
            if variant_config["use_consistency_loss"] and model.refined_stage is not None:
                c_loss = consistency_loss(out_first, out_refined)
                loss += 0.1 * c_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}: Avg Loss = {epoch_loss / len(dataloader):.4f}")
    
    return model

def train_tg_gcve(data_dir, config_path=None, variant=None):
    """
    Main function to train the TG-GCVE model.
    
    Args:
        data_dir (str): Directory containing training data
        config_path (str, optional): Path to configuration file
        variant (str, optional): Variant of the model to train
        
    Returns:
        nn.Module: Trained model
    """
    print(f"Training TG-GCVE model with data from {data_dir}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config = {
        'batch_size': 2,
        'frame_size': (256, 256),
        'sequence_length': 16,
        'epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'noise_level': 0.1,
        'perceptual_weight': 0.1,
        'consistency_weight': 0.1,
        'global_editing_intensity': 1.0,
        'print_freq': 10,
        'save_freq': 1,
        'target_text': 'edited video',
        'use_consistency_loss': True,
        'checkpoint_dir': 'models'
    }
    
    # Update config from file if provided
    if config_path is not None and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created directory: {data_dir}")
    
    # Preprocess data
    train_dataloader = preprocess_for_training(
        data_dir=data_dir,
        batch_size=config['batch_size'],
        frame_size=config['frame_size'],
        sequence_length=config['sequence_length']
    )
    
    # Define model variants
    variants = {
        "full_TG_GCVE": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": True
        },
        "single_stage": {
            "use_refined_stage": False, 
            "use_consistency_loss": False, 
            "use_global_context": True
        },
        "no_consistency": {
            "use_refined_stage": True, 
            "use_consistency_loss": False, 
            "use_global_context": True
        },
        "no_global_context": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": False
        },
        "iterative_refinement": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": True,
            "iterative_refinement": True,
            "num_iterations": 3
        }
    }
    
    # Select variant to train
    if variant is not None and variant in variants:
        variant_config = variants[variant]
        print(f"Training variant: {variant}")
    else:
        # Default to full TG-GCVE
        variant_config = variants["full_TG_GCVE"]
        print("Training default variant: full_TG_GCVE")
    
    # Create model
    model = TGGCVEModule(
        in_channels=3,
        hidden_dim=64,
        use_refined_stage=variant_config["use_refined_stage"],
        use_consistency_loss=variant_config["use_consistency_loss"],
        use_global_context=variant_config["use_global_context"],
        iterative_refinement=variant_config.get("iterative_refinement", False),
        num_iterations=variant_config.get("num_iterations", 3)
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Create LPIPS loss function
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Load CLIP model for validation
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    train_losses = []
    
    for epoch in range(config['epochs']):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            lpips_loss_fn=lpips_loss_fn,
            epoch=epoch,
            config=config
        )
        train_losses.append(train_loss)
        
        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'config': config,
                    'variant': variant
                },
                os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            )
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(config['checkpoint_dir'], 'final_model.pth')
    )
    print("Training completed. Saved final model.")
    
    return model

# --------------------------
# Experiment Functions
# --------------------------

def experiment_consistency_loss_ablation(data_dir, epochs=2):
    """
    Experiment 1: Ablation study on Tweedie-inspired consistency loss.
    
    Args:
        data_dir (str): Directory containing training data
        epochs (int): Number of epochs to train each variant
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\n--- Experiment 1: Consistency Loss Ablation ---")
    
    # Preprocess data
    dataloader = preprocess_for_training(
        data_dir=data_dir,
        batch_size=4,
        frame_size=(256, 256),
        sequence_length=16
    )
    
    # Define variants
    variants = {
        "full_TG_GCVE": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": True
        },
        "single_stage": {
            "use_refined_stage": False, 
            "use_consistency_loss": False, 
            "use_global_context": True
        },
        "no_consistency": {
            "use_refined_stage": True, 
            "use_consistency_loss": False, 
            "use_global_context": True
        }
    }
    
    # Train each variant
    models = {}
    for name, config in variants.items():
        print(f"\n---- Starting variant: {name} ----")
        model = train_variant(config, dataloader, epochs=epochs)
        models[name] = model
    
    return models

def experiment_global_context_evaluation(data_dir, epochs=2):
    """
    Experiment 2: Evaluation of global context aggregation for global editing.
    
    Args:
        data_dir (str): Directory containing training data
        epochs (int): Number of epochs to train each variant
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\n--- Experiment 2: Global Context Aggregation Evaluation ---")
    
    # Preprocess data
    dataloader = preprocess_for_training(
        data_dir=data_dir,
        batch_size=4,
        frame_size=(256, 256),
        sequence_length=16
    )
    
    # Define variants
    variants = {
        "with_global_context": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": True
        },
        "without_global_context": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": False
        }
    }
    
    # Train each variant
    models = {}
    for name, config in variants.items():
        print(f"\n---- Starting variant: {name} ----")
        model = train_variant(config, dataloader, epochs=epochs)
        models[name] = model
    
    return models

def experiment_adaptive_fusion_refinement(data_dir, epochs=2):
    """
    Experiment 3: Evaluation of adaptive fusion and iterative refinement.
    
    Args:
        data_dir (str): Directory containing training data
        epochs (int): Number of epochs to train each variant
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\n--- Experiment 3: Adaptive Fusion and Iterative Refinement ---")
    
    # Preprocess data
    dataloader = preprocess_for_training(
        data_dir=data_dir,
        batch_size=4,
        frame_size=(256, 256),
        sequence_length=16
    )
    
    # Define variants
    variants = {
        "standard": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": True,
            "iterative_refinement": False
        },
        "with_refinement": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": True,
            "iterative_refinement": True,
            "num_iterations": 3
        }
    }
    
    # Train each variant
    models = {}
    for name, config in variants.items():
        print(f"\n---- Starting variant: {name} ----")
        model = train_variant(config, dataloader, epochs=epochs)
        models[name] = model
    
    return models

def visualize_global_context(model, sample_input):
    """
    Visualize the global context attention maps.
    
    Args:
        model (nn.Module): Trained model with global context module
        sample_input (torch.Tensor): Sample input tensor
        
    Returns:
        None (displays visualization)
    """
    model.eval()
    with torch.no_grad():
        # Forward pass
        _, _, _, attn_map, _ = model(sample_input)
        
        if attn_map is not None:
            # Convert attention map to numpy for visualization
            attn_np = attn_map[0].cpu().numpy()
            
            # Normalize for visualization
            attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
            
            # Convert input to numpy for visualization
            input_np = sample_input[0].permute(1, 2, 0).cpu().numpy()
            input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)
            
            # Create figure
            plt.figure(figsize=(12, 5))
            
            # Plot input image
            plt.subplot(1, 2, 1)
            plt.imshow(input_np)
            plt.title("Input Image")
            plt.axis("off")
            
            # Plot attention map
            plt.subplot(1, 2, 2)
            plt.imshow(attn_np, cmap="viridis")
            plt.title("Global Context Attention Map")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            
            print("Global context visualization complete.")
        else:
            print("No attention map available. Model may not have global context module.")

# --------------------------
# Main Function
# --------------------------

if __name__ == "__main__":
    # Example usage
    data_dir = "data/sample_videos"
    
    # Check if data directory exists, if not create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created directory: {data_dir}")
        print("Please add video files to this directory for training")
    
    # If data directory exists and contains video files, train the model
    elif any(f.endswith(('.mp4', '.avi', '.mov')) for f in os.listdir(data_dir)):
        # Train the model
        model = train_tg_gcve(
            data_dir=data_dir,
            variant="full_TG_GCVE"
        )
        
        print("Model training completed successfully.")
    else:
        print(f"No video files found in {data_dir}")
        print("Please add video files (mp4, avi, mov) to this directory for training")
