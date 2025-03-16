"""
Model training module for ACM optimizer experiments.

This module implements training functions for:
1. Synthetic function optimization
2. CIFAR-10 image classification model training
3. Ablation studies
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.optimizers import ACM, ACM_NoCurvature, ACM_NoRegularization

class SimpleCNN(nn.Module):
    """
    A simple CNN model for CIFAR-10 classification.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_synthetic(func, init_params, optimizer_class, optim_kwargs, 
                   num_iterations=5000, loss_threshold=1e-3, log_interval=200):
    """
    Train on synthetic optimization functions.
    
    Args:
        func (callable): The function to optimize
        init_params (list): Initial parameters
        optimizer_class: Optimizer class to use
        optim_kwargs (dict): Optimizer parameters
        num_iterations (int): Maximum number of iterations
        loss_threshold (float): Early stopping threshold
        log_interval (int): Interval for logging progress
        
    Returns:
        tuple: (trajectory, losses)
    """
    print(f"Running synthetic experiment with {optimizer_class.__name__}")
    traj = []  # Parameter trajectories
    losses = []  # Loss values
    
    # Create the parameter vector (with gradients enabled)
    params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    optimizer = optimizer_class([params], **optim_kwargs)
    
    for itr in range(num_iterations):
        optimizer.zero_grad()
        loss = func(params)
        loss.backward()
        optimizer.step()
        
        # Record trajectory and loss
        traj.append(params.data.clone().numpy())
        losses.append(loss.item())
        
        # Log progress
        if itr % log_interval == 0:
            print(f"Iteration {itr}: Loss = {loss.item():.4e}, Params = {params.data.numpy()}")
        
        # Check for convergence
        if loss.item() < loss_threshold:
            print(f"Converged at iteration {itr} with loss {loss.item():.4e}")
            break
    
    return np.array(traj), losses

def train_cifar10(model, train_loader, val_loader, optimizer_class, optim_kwargs, 
                 num_epochs=10, device=None, save_dir='./models', experiment_name='cifar10'):
    """
    Train a model on CIFAR-10 dataset.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer_class: Optimizer class to use
        optim_kwargs (dict): Optimizer parameters
        num_epochs (int): Number of training epochs
        device (torch.device): Device to use for training
        save_dir (str): Directory to save model checkpoints
        experiment_name (str): Name for the experiment
        
    Returns:
        dict: Training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on {device}")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), **optim_kwargs)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * images.size(0)
            
            # Print batch progress
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        total_samples = 0
        for i, (images, _) in enumerate(train_loader):
            if hasattr(train_loader, 'dataset'):
                total_samples = len(train_loader.dataset)
                break
            else:
                # For custom samplers without dataset attribute
                total_samples += images.size(0)
                if i >= len(train_loader) - 1:
                    break
        
        train_loss = running_loss / total_samples
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / total
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(save_dir, f"{experiment_name}_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{experiment_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return history

def run_ablation_study(func, init_params, num_iterations=2000, loss_threshold=1e-5):
    """
    Run ablation study comparing different ACM variants.
    
    Args:
        func (callable): The function to optimize
        init_params (list): Initial parameters
        num_iterations (int): Maximum number of iterations
        loss_threshold (float): Early stopping threshold
        
    Returns:
        dict: Results for each optimizer variant
    """
    print("\n======== Ablation Study Experiment ========")
    
    # Define optimizer configurations
    optimizers_ablation = {
        'Full ACM': (ACM, {'lr': 1e-3, 'betas': (0.9, 0.999), 'curvature_coeff': 1e-2}),
        'ACM_NoCurvature': (ACM_NoCurvature, {'lr': 1e-3, 'betas': (0.9, 0.999)}),
        'ACM_NoRegularization': (ACM_NoRegularization, {'lr': 1e-3, 'betas': (0.9, 0.999), 'curvature_coeff': 1e-2})
    }
    
    results = {}
    
    for name, (opt_class, kwargs) in optimizers_ablation.items():
        print(f"\nTesting {name} on function")
        traj, losses = train_synthetic(
            func, init_params, opt_class, kwargs,
            num_iterations=num_iterations, 
            loss_threshold=loss_threshold
        )
        results[name] = {'trajectory': traj, 'losses': losses}
        print(f"{name}: {len(losses)} iterations; final loss: {losses[-1]:.4e}")
    
    return results
