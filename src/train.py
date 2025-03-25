"""
Training module for MCAD experiments.
Implements model architecture and training procedures.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time

class SimpleDenoiser(nn.Module):
    """
    Simple denoising model for MCAD experiments.
    """
    def __init__(self, base_channels=32):
        super(SimpleDenoiser, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the model."""
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class MCADTrainer:
    """
    Trainer class for MCAD method.
    Handles model training with momentum and adaptive consistency.
    """
    def __init__(self, model, optimizer, device, use_momentum=True, adaptive_consistency=True, momentum_beta=0.9):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_momentum = use_momentum
        self.adaptive_consistency = adaptive_consistency
        self.momentum_beta = momentum_beta
        self.momentum_state = {}
        self.mse_loss = nn.MSELoss()
    
    def consistency_loss(self, reconstruction, target, noise_level=0.1):
        """
        Calculate consistency loss with optional adaptive weighting.
        """
        if self.adaptive_consistency: 
            # Adaptive weighting based on noise level
            weight = 1.0 - (noise_level * 0.5)
        else:
            weight = 1.0
            
        return weight * self.mse_loss(reconstruction, target)
    
    def train_batch(self, data, noise_std=0.1):
        """Train the model on a single batch."""
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        
        # Apply noise to simulate corruption
        corrupted = data + torch.randn_like(data) * noise_std
        
        # Forward pass (first Tweedie step)
        initial_reconstruction = self.model(corrupted)
        
        # Compute initial loss
        loss = self.consistency_loss(initial_reconstruction, data, noise_std)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply momentum update if enabled
        if self.use_momentum:
            for p in self.model.parameters():
                if p.grad is not None:
                    # Update momentum state with beta factor
                    self.momentum_state[p] = self.momentum_beta * self.momentum_state.get(p, 0) + p.grad
                    # Replace gradient with momentum-corrected gradient
                    p.grad = self.momentum_state[p]
                    
        # Update model parameters
        self.optimizer.step()
        
        # Second Tweedie pass with momentum-corrected model (if using momentum)
        if self.use_momentum:
            # This simulates the "Fast-Tweedie Inversion" described in the method
            final_reconstruction = self.model(corrupted)
            # We don't need to backpropagate through this, just for monitoring
            final_loss = self.mse_loss(final_reconstruction, data).item()
        else:
            final_loss = loss.item()
            
        return loss.item(), final_loss

def train_epoch(model_trainer, dataloader, config, max_batches=None, log=True):
    """Train the model for one epoch."""
    running_loss = 0.0
    running_final_loss = 0.0
    start_time = time.time()
    
    for idx, (data, _) in enumerate(dataloader):
        # Break after max_batches if specified (for test runs)
        if max_batches is not None and idx >= max_batches:
            break
            
        # Train on batch
        loss, final_loss = model_trainer.train_batch(
            data, 
            noise_std=config['training']['noise_std']
        )
        
        # Update running losses
        running_loss += loss
        running_final_loss += final_loss
        
        # Log progress
        if log and (idx+1) % 100 == 0:
            print(f"  Batch {idx+1}: Loss {loss:.4f}, Final Loss {final_loss:.4f}")
    
    # Calculate average losses
    batch_count = min(len(dataloader), max_batches if max_batches is not None else float('inf'))
    avg_loss = running_loss / batch_count if batch_count > 0 else 0
    avg_final_loss = running_final_loss / batch_count if batch_count > 0 else 0
    
    # Calculate epoch time
    epoch_time = time.time() - start_time
    
    return {
        'avg_loss': avg_loss,
        'avg_final_loss': avg_final_loss,
        'epoch_time': epoch_time,
        'batch_count': batch_count
    }

def create_model(config, device):
    """Create and initialize model based on configuration."""
    model = SimpleDenoiser(base_channels=config['model']['base_channels']).to(device)
    return model

def create_trainer(model, config, device):
    """Create optimizer and trainer."""
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    trainer = MCADTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_momentum=config['model']['use_momentum'],
        adaptive_consistency=config['model']['adaptive_consistency'],
        momentum_beta=config['training']['momentum_beta']
    )
    return trainer
