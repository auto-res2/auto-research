"""Training module for the ACM optimizer experiment."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from tqdm import tqdm

from .utils.optimizers import ACMOptimizer
from config.experiment_config import OPTIMIZER_CONFIG


def train_model(model, train_loader, optimizer_name, device, epochs, log_interval=100):
    """Train a model using the specified optimizer.
    
    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): DataLoader for training data
        optimizer_name (str): Name of the optimizer to use ('acm', 'adam', or 'sgd_mom')
        device (torch.device): Device to train on
        epochs (int): Number of epochs to train for
        log_interval (int): How often to log progress
        
    Returns:
        dict: Dictionary containing training results (losses, accuracies)
    """
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer based on optimizer_name
    if optimizer_name == 'acm':
        optimizer = ACMOptimizer(
            model.parameters(),
            lr=OPTIMIZER_CONFIG['acm']['lr'],
            beta=OPTIMIZER_CONFIG['acm']['beta'],
            curvature_influence=OPTIMIZER_CONFIG['acm']['curvature_influence']
        )
    elif optimizer_name == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=OPTIMIZER_CONFIG['adam']['lr'],
            betas=OPTIMIZER_CONFIG['adam']['betas']
        )
    elif optimizer_name == 'sgd_mom':
        optimizer = SGD(
            model.parameters(),
            lr=OPTIMIZER_CONFIG['sgd_mom']['lr'],
            momentum=OPTIMIZER_CONFIG['sgd_mom']['momentum']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Training loop
    model.train()
    train_losses = []
    train_accs = []
    
    print(f"Starting training with {optimizer_name} optimizer")
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                accuracy = 100.0 * correct / total
                
                train_losses.append(avg_loss)
                train_accs.append(accuracy)
                
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'acc': f"{accuracy:.2f}%"
                })
                
                running_loss = 0.0
                correct = 0
                total = 0
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return {
        'losses': train_losses,
        'accuracies': train_accs,
        'time': end_time - start_time
    }


def run_synthetic_experiment(optimizer_name, num_iters=100):
    """Run a synthetic optimization experiment.
    
    Args:
        optimizer_name (str): Name of the optimizer to use ('acm', 'adam', or 'sgd_mom')
        num_iters (int): Number of iterations to run
        
    Returns:
        dict: Dictionary containing experiment results
    """
    print(f"=== Synthetic Experiment with {optimizer_name} ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Example 2D quadratic. A is positive definite.
    A = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
    b = torch.tensor([1.0, 1.0])
    
    # Initialize parameters to optimize
    x_data = torch.randn(2, requires_grad=True)
    
    # Define quadratic loss function
    def quadratic_loss(x, A, b):
        return 0.5 * x @ A @ x - b @ x
    
    # Initialize optimizer based on optimizer_name
    if optimizer_name == 'acm':
        optimizer = ACMOptimizer(
            [x_data],
            lr=OPTIMIZER_CONFIG['acm']['lr'],
            beta=OPTIMIZER_CONFIG['acm']['beta'],
            curvature_influence=OPTIMIZER_CONFIG['acm']['curvature_influence']
        )
    elif optimizer_name == 'adam':
        optimizer = Adam(
            [x_data],
            lr=OPTIMIZER_CONFIG['adam']['lr'],
            betas=OPTIMIZER_CONFIG['adam']['betas']
        )
    elif optimizer_name == 'sgd_mom':
        optimizer = SGD(
            [x_data],
            lr=OPTIMIZER_CONFIG['sgd_mom']['lr'],
            momentum=OPTIMIZER_CONFIG['sgd_mom']['momentum']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Optimization loop
    losses = []
    trajectories = []
    
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = quadratic_loss(x_data, A, b)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        trajectories.append(x_data.detach().clone().numpy())
        
        if (i + 1) % (num_iters // 5) == 0 or i == 0:
            print(f"Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
    
    return {
        'losses': losses,
        'trajectories': np.array(trajectories)
    }
