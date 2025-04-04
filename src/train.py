"""
Training implementations for SAC-Seg experiments.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union

from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    seed_config: Optional[Dict] = None
) -> Tuple[float, float, float]:
    """
    Train a model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (CPU or GPU)
        seed_config: Optional configuration for seed-based methods
        
    Returns:
        Tuple of (epoch_loss, epoch_time, peak_memory_mb)
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)
        
        if hasattr(model, 'seed_config') and seed_config is not None:
            model.seed_config = seed_config
            
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(dataloader)
    
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        peak_memory_mb = 0.0
        
    return epoch_loss, epoch_time, peak_memory_mb


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    metric_fn: Callable,
    device: torch.device,
    seed_config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        metric_fn: Function to compute evaluation metrics
        device: Device to use (CPU or GPU)
        seed_config: Optional configuration for seed-based methods
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics_sum = {}
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images, targets = images.to(device), targets.to(device)
            
            if hasattr(model, 'seed_config') and seed_config is not None:
                model.seed_config = seed_config
                
            outputs = model(images)
            batch_metrics = metric_fn(outputs, targets)
            
            if not metrics_sum:
                metrics_sum = {k: 0.0 for k in batch_metrics.keys()}
                
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
    
    metrics_avg = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    return metrics_avg


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    metric_fn: Callable,
    device: torch.device,
    num_epochs: int,
    seed_config: Optional[Dict] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, List]:
    """
    Train a model for multiple epochs with validation.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        metric_fn: Function to compute evaluation metrics
        device: Device to use (CPU or GPU)
        num_epochs: Number of epochs to train
        seed_config: Optional configuration for seed-based methods
        scheduler: Optional learning rate scheduler
        
    Returns:
        Dictionary of training history
    """
    history = {
        'train_loss': [],
        'val_metrics': [],
        'epoch_times': [],
        'peak_memory_mb': []
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss, epoch_time, peak_memory = train_one_epoch(
            model, train_loader, criterion, optimizer, device, seed_config
        )
        
        val_metrics = evaluate(model, val_loader, metric_fn, device, seed_config)
        
        if scheduler is not None:
            scheduler.step()
            
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        history['epoch_times'].append(epoch_time)
        history['peak_memory_mb'].append(peak_memory)
        
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        print(f"Train Loss: {train_loss:.4f}, {metrics_str}")
        print(f"Epoch Time: {epoch_time:.2f}s, Peak Memory: {peak_memory:.2f} MB")
        print("-" * 60)
        
    return history
