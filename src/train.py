"""
Training module for DITTO-GSD experiments.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from pytorch3d.loss import chamfer_distance
except ImportError:
    print("pytorch3d not available, using custom chamfer distance implementation")
    
    def chamfer_distance(x, y):
        """
        Custom implementation of chamfer distance without pytorch3d.
        
        Args:
            x: First point cloud (B, N, 3)
            y: Second point cloud (B, N, 3)
            
        Returns:
            Tuple of (loss, _)
        """
        x_expanded = x.unsqueeze(2)  # (B, N, 1, 3)
        y_expanded = y.unsqueeze(1)  # (B, 1, N, 3)
        
        dist = torch.sum((x_expanded - y_expanded) ** 2, dim=-1)  # (B, N, N)
        
        x_to_y = torch.min(dist, dim=2)[0]  # (B, N)
        y_to_x = torch.min(dist, dim=1)[0]  # (B, N)
        
        loss = torch.mean(x_to_y) + torch.mean(y_to_x)
        
        return loss, None

def train_model(model, dataloader, num_epochs=5, device='cuda', 
                use_geo_loss=False, lr=1e-4, weight_decay=1e-5):
    """
    Train a model.
    
    Args:
        model: Model to train.
        dataloader: Dataloader for training data.
        num_epochs: Number of epochs to train.
        device: Device to use for training.
        use_geo_loss: Whether to use geometric distillation loss.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        
    Returns:
        Trained model and loss history.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    model.train()
    loss_history = []
    
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for pc, gt in dataloader:
            pc = pc.to(device)
            gt = gt.to(device)
            
            optimizer.zero_grad()
            pred = model(pc)
            
            loss = loss_fn(pred, gt)
            
            chamfer_loss, _ = chamfer_distance(pred.unsqueeze(0), gt.unsqueeze(0))
            loss = loss + chamfer_loss
            
            if use_geo_loss:
                geo_loss = torch.mean((pred - gt)**2) * 0.1
                loss = loss + geo_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model, loss_history

def train_variant(config, dataset, epochs=5, device='cuda'):
    """
    Train a model variant using the configuration dictionary.
    
    Args:
        config: Configuration dictionary.
        dataset: Dataset to train on.
        epochs: Number of epochs to train.
        device: Device to use for training.
        
    Returns:
        Loss history for each epoch.
    """
    from utils.models import DITTOModel, DITTOGSDModel
    from preprocess import preprocess_data
    
    if config["use_gs_decoder"]:
        model = DITTOGSDModel(use_proj=config.get("use_proj", True))
    else:
        model = DITTOModel()
    
    dataloader = preprocess_data(dataset)
    
    _, loss_history = train_model(
        model, 
        dataloader, 
        num_epochs=epochs, 
        device=device,
        use_geo_loss=config.get("use_geo_loss", False)
    )
    
    return loss_history
