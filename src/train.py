import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter

class GSDModel(nn.Module):
    """
    Geometric Score Distillation (GSD) model for one-step disentangled diffusion
    """
    def __init__(self, config):
        super(GSDModel, self).__init__()
        self.config = config
        channels = config['model']['channels']
        
        # Encoder network (mapping from image space to latent space)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),  # 256->128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),       # 128->64
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),      # 64->32
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),      # 32->16
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, config['model']['latent_dim'], kernel_size=4, stride=2, padding=1)  # 16->8
        )
        
        # Decoder network (mapping from latent space to image space)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config['model']['latent_dim'], 512, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64->128
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),  # 128->256
            nn.Tanh()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

def isometry_loss(latent1, latent2, x1, x2):
    """
    Compute the isometry loss to enforce the preservation of distances
    between the image space and the latent space
    """
    # Compute pairwise Euclidean distances in the image space
    x_dist = torch.norm(x1.reshape(x1.size(0), -1) - x2.reshape(x2.size(0), -1), dim=1)
    
    # Compute pairwise Euclidean distances in the latent space
    latent_dist = torch.norm(latent1.reshape(latent1.size(0), -1) - latent2.reshape(latent2.size(0), -1), dim=1)
    
    # The isometry loss enforces that these distances are proportional
    return torch.mean((x_dist - latent_dist)**2)

def score_identity_loss(output, target):
    """
    Compute the score identity loss for distillation
    """
    return torch.mean((output - target)**2)

def train_model(config, train_loader, val_loader=None, checkpoint_path=None):
    """
    Train the GSD model
    
    Args:
        config: Configuration dictionary
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        checkpoint_path: Path to save model checkpoints (optional)
        
    Returns:
        model: Trained GSD model
    """
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GSDModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create directory for checkpoints if it doesn't exist
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir='logs')
    
    # Training loop
    best_val_loss = float('inf')
    num_epochs = config['training']['epochs']
    lambda_iso = config['training']['lambda_iso']
    lambda_score = config['training']['lambda_score']
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_iso_loss = 0.0
        train_score_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Get a permuted batch for pairwise distances
            indices = torch.randperm(data.size(0))
            data_perm = data[indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, latent = model(data)
            recon_perm, latent_perm = model(data_perm)
            
            # Compute losses
            iso_loss = isometry_loss(latent, latent_perm, data, data_perm)
            rec_loss = score_identity_loss(recon, data)
            
            # Combine losses
            total_loss = lambda_iso * iso_loss + lambda_score * rec_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += total_loss.item()
            train_iso_loss += iso_loss.item()
            train_score_loss += rec_loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} | Iso Loss: {iso_loss.item():.4f} | "
                      f"Score Loss: {rec_loss.item():.4f} | Total Loss: {total_loss.item():.4f}")
            
            # Limit training for test run
            if config.get('test_run', False) and batch_idx >= 1:
                break
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iso_loss = train_iso_loss / len(train_loader)
        avg_train_score_loss = train_score_loss / len(train_loader)
        
        # Log training losses
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_iso', avg_train_iso_loss, epoch)
        writer.add_scalar('Loss/train_score', avg_train_score_loss, epoch)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_iso_loss = 0.0
            val_score_loss = 0.0
            
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    
                    # Get a permuted batch for pairwise distances
                    indices = torch.randperm(data.size(0))
                    data_perm = data[indices]
                    
                    # Forward pass
                    recon, latent = model(data)
                    recon_perm, latent_perm = model(data_perm)
                    
                    # Compute losses
                    iso_loss = isometry_loss(latent, latent_perm, data, data_perm)
                    rec_loss = score_identity_loss(recon, data)
                    
                    # Combine losses
                    total_loss = lambda_iso * iso_loss + lambda_score * rec_loss
                    
                    # Update statistics
                    val_loss += total_loss.item()
                    val_iso_loss += iso_loss.item()
                    val_score_loss += rec_loss.item()
                    
                    # Limit validation for test run
                    if config.get('test_run', False):
                        break
            
            # Calculate average losses
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iso_loss = val_iso_loss / len(val_loader)
            avg_val_score_loss = val_score_loss / len(val_loader)
            
            # Log validation losses
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Loss/val_iso', avg_val_iso_loss, epoch)
            writer.add_scalar('Loss/val_score', avg_val_score_loss, epoch)
            
            # Save checkpoint if validation loss improves
            if avg_val_loss < best_val_loss and checkpoint_path is not None:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved model checkpoint at epoch {epoch}")
        
        # Print epoch summary
        time_per_epoch = time.time() - start_time
        print(f"Epoch {epoch} completed in {time_per_epoch:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Iso Loss: {avg_train_iso_loss:.4f} | "
              f"Train Score Loss: {avg_train_score_loss:.4f}")
        
        if val_loader is not None:
            print(f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Iso Loss: {avg_val_iso_loss:.4f} | "
                  f"Val Score Loss: {avg_val_score_loss:.4f}")
        
        # Early stopping for test run
        if config.get('test_run', False) and epoch >= 0:
            break
    
    writer.close()
    
    # Save final model
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path.replace('.pt', '_final.pt'))
        print(f"Saved final model")
    
    return model
