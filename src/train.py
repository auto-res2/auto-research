# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from utils.models import AmbientDiffusionModel, OneStepGenerator
from utils.data import get_dataloaders

def train_ambient_diffusion(
    noise_level=0.3,
    batch_size=64,
    epochs=1,
    lr=1e-4,
    device='cuda',
    save_dir='./models'
):
    """
    Train the ambient diffusion model on noisy data.
    
    Args:
        noise_level: Level of noise in the data
        batch_size: Batch size for training
        epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save the trained model
        
    Returns:
        model: Trained ambient diffusion model
    """
    print(f"Training ambient diffusion model with noise level {noise_level}...")
    
    # Create model directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, _ = get_dataloaders(batch_size=batch_size, noise_level=noise_level)
    
    # Initialize model
    model = AmbientDiffusionModel().to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize loss function
    mse_loss = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (noisy_imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            noisy_imgs = noisy_imgs.to(device)
            
            # Target for ambient diffusion is less noisy images
            # We simulate this by using a fixed step for demonstration
            target_step = 30
            with torch.no_grad():
                targets = model(noisy_imgs, step=target_step)
            
            # Forward pass
            outputs = model(noisy_imgs, step=0)
            
            # Compute loss (consistency loss)
            loss = mse_loss(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # For test purposes, break after a few batches
            if batch_idx >= 5:
                break
        
        avg_loss = train_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    model_path = os.path.join(save_dir, f"ambient_diffusion_noise_{noise_level}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

def train_one_step_generator(
    teacher_model,
    noise_level=0.3,
    batch_size=64,
    epochs=1,
    lr=1e-4,
    device='cuda',
    save_dir='./models'
):
    """
    Train the one-step generator using ambient score distillation.
    
    Args:
        teacher_model: Trained ambient diffusion model
        noise_level: Level of noise in the data
        batch_size: Batch size for training
        epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save the trained model
        
    Returns:
        model: Trained one-step generator
    """
    print(f"Training one-step generator with score distillation...")
    
    # Create model directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, _ = get_dataloaders(batch_size=batch_size, noise_level=noise_level)
    
    # Initialize model
    model = OneStepGenerator().to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize loss function
    mse_loss = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        score_id_losses = []
        consistency_losses = []
        
        for batch_idx, (noisy_imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            noisy_imgs = noisy_imgs.to(device)
            
            # Generate target scores from teacher model
            with torch.no_grad():
                teacher_scores = teacher_model(noisy_imgs, step=25)
            
            # Forward pass
            student_scores = model(noisy_imgs)
            
            # Compute score identity loss
            score_id_loss = mse_loss(student_scores, teacher_scores)
            
            # Compute consistency loss (optional)
            # Simulate by adding small perturbation and checking if output remains similar
            perturbation = torch.randn_like(noisy_imgs) * 0.05
            perturbed_output = model(noisy_imgs + perturbation)
            consistency_loss = mse_loss(perturbed_output, student_scores)
            
            # Total loss
            loss = score_id_loss + 0.1 * consistency_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            score_id_losses.append(score_id_loss.item())
            consistency_losses.append(consistency_loss.item())
            
            # For test purposes, break after a few batches
            if batch_idx >= 5:
                break
        
        avg_loss = train_loss / (batch_idx + 1)
        avg_score_id_loss = np.mean(score_id_losses)
        avg_consistency_loss = np.mean(consistency_losses)
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Average Loss: {avg_loss:.4f}, "
              f"Score ID Loss: {avg_score_id_loss:.4f}, "
              f"Consistency Loss: {avg_consistency_loss:.4f}")
    
    # Save model
    model_path = os.path.join(save_dir, f"one_step_generator_noise_{noise_level}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    # Simple test run of the training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    ambient_model = train_ambient_diffusion(noise_level=0.3, epochs=1, device=device)
    one_step_model = train_one_step_generator(ambient_model, noise_level=0.3, epochs=1, device=device)
