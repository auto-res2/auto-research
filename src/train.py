"""
Training script for the PTDA model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.ptda_model import PTDAModel, AblatedPTDAModel, BaselineModel
from src.utils.data import VideoFrameDataset, get_dataloader
from src.utils.visualization import visualize_latent_space
from config.ptda.config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS


def train_model(model, dataloader, optimizer, criterion, device='cuda', epochs=None, save_dir=None, save_interval=None, kl_weight=0.01):
    """
    Train a model.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on
        epochs: Number of epochs to train for
        save_dir: Directory to save model checkpoints
        save_interval: Interval to save model checkpoints
        kl_weight: Weight for KL divergence loss
        
    Returns:
        List of latent vectors (if model has latent branch)
    """
    if epochs is None:
        epochs = TRAINING_CONFIG['num_epochs']
    
    if save_interval is None:
        save_interval = TRAINING_CONFIG['save_interval']
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    latent_vectors = []
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'include_latent') and model.include_latent:
                features, mu, logvar = model.encode(batch[:, 0])
                latent = model.reparameterize(mu, logvar)
                reconstruction = model.decode(features, latent)
                
                recon_loss = criterion(reconstruction, batch[:, 0])
                kl_loss = model.compute_kl_loss(mu, logvar)
                loss = recon_loss + kl_weight * kl_loss
                
                latent_vectors.append(latent.detach().cpu())
            else:
                reconstruction = model(batch[:, 0])
                loss = criterion(reconstruction, batch[:, 0])
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        if save_dir is not None and (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
    
    if save_dir is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        loss_plot_path = os.path.join(save_dir, "training_loss.pdf")
        plt.savefig(loss_plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training loss plot to {loss_plot_path}")
    
    return latent_vectors


def train_ptda_model(device='cuda'):
    """
    Train the PTDA model.
    
    Args:
        device: Device to train on
        
    Returns:
        Trained model
    """
    print("Training PTDA model...")
    
    model = PTDAModel(
        include_latent=MODEL_CONFIG['include_latent'],
        latent_dim=MODEL_CONFIG['latent_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    dataset = VideoFrameDataset(
        num_samples=DATA_CONFIG['num_frames'],
        num_frames=DATA_CONFIG['num_frames'],
        height=DATA_CONFIG['frame_height'],
        width=DATA_CONFIG['frame_width']
    )
    dataloader = get_dataloader(
        dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    criterion = nn.MSELoss()
    
    latent_vectors = train_model(
        model,
        dataloader,
        optimizer,
        criterion,
        device=device,
        epochs=TRAINING_CONFIG['num_epochs'],
        save_dir=PATHS['models_dir'],
        save_interval=TRAINING_CONFIG['save_interval'],
        kl_weight=TRAINING_CONFIG['kl_weight']
    )
    
    if latent_vectors:
        os.makedirs(PATHS['figures_dir'], exist_ok=True)
        visualize_latent_space(
            latent_vectors,
            "PTDA Latent Space",
            os.path.join(PATHS['figures_dir'], "ptda_latent_space.pdf")
        )
    
    return model


def train_ablated_model(device='cuda'):
    """
    Train the ablated PTDA model.
    
    Args:
        device: Device to train on
        
    Returns:
        Trained model
    """
    print("Training ablated PTDA model...")
    
    model = AblatedPTDAModel(
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    dataset = VideoFrameDataset(
        num_samples=DATA_CONFIG['num_frames'],
        num_frames=DATA_CONFIG['num_frames'],
        height=DATA_CONFIG['frame_height'],
        width=DATA_CONFIG['frame_width']
    )
    dataloader = get_dataloader(
        dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    criterion = nn.MSELoss()
    
    train_model(
        model,
        dataloader,
        optimizer,
        criterion,
        device=device,
        epochs=TRAINING_CONFIG['num_epochs'],
        save_dir=os.path.join(PATHS['models_dir'], 'ablated'),
        save_interval=TRAINING_CONFIG['save_interval']
    )
    
    return model


def train_baseline_model(device='cuda'):
    """
    Train the baseline model.
    
    Args:
        device: Device to train on
        
    Returns:
        Trained model
    """
    print("Training baseline model...")
    
    model = BaselineModel(
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    dataset = VideoFrameDataset(
        num_samples=DATA_CONFIG['num_frames'],
        num_frames=DATA_CONFIG['num_frames'],
        height=DATA_CONFIG['frame_height'],
        width=DATA_CONFIG['frame_width']
    )
    dataloader = get_dataloader(
        dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    criterion = nn.MSELoss()
    
    train_model(
        model,
        dataloader,
        optimizer,
        criterion,
        device=device,
        epochs=TRAINING_CONFIG['num_epochs'],
        save_dir=os.path.join(PATHS['models_dir'], 'baseline'),
        save_interval=TRAINING_CONFIG['save_interval']
    )
    
    return model


def main():
    """
    Main function for model training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    os.makedirs(PATHS['logs_dir'], exist_ok=True)
    
    ptda_model = train_ptda_model(device=device)
    ablated_model = train_ablated_model(device=device)
    baseline_model = train_baseline_model(device=device)
    
    print("Model training completed.")


if __name__ == "__main__":
    main()
