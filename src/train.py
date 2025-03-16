"""
Training module for ACM optimizer experiments.

This module implements the training procedures for comparing the Adaptive Curvature
Momentum (ACM) optimizer with other standard optimizers.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18

from src.optimizers import ACMOptimizer, AdaBelief


class SimpleCNN(nn.Module):
    """
    A simple CNN model for CIFAR-10 classification.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class QuadraticModel(nn.Module):
    """
    A simple model for the quadratic function experiment.
    """
    def __init__(self, input_dim=10):
        super(QuadraticModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x)


def get_optimizer(optimizer_name, model_params, config):
    """
    Create an optimizer based on the configuration.
    
    Args:
        optimizer_name (str): Name of the optimizer
        model_params (iterable): Model parameters to optimize
        config (dict): Configuration dictionary with optimizer parameters
        
    Returns:
        optimizer: PyTorch optimizer
    """
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    
    elif optimizer_name == 'adabelief':
        return AdaBelief(model_params, lr=lr, weight_decay=weight_decay)
    
    elif optimizer_name == 'acm':
        curvature_coef = config.get('curvature_coef', 0.1)
        return ACMOptimizer(
            model_params, lr=lr, weight_decay=weight_decay,
            curvature_coef=curvature_coef
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_model(model_name, config):
    """
    Create a model based on the configuration.
    
    Args:
        model_name (str): Name of the model
        config (dict): Configuration dictionary with model parameters
        
    Returns:
        model: PyTorch model
    """
    if model_name == 'resnet18':
        model = resnet18(num_classes=config.get('num_classes', 10))
        return model
    
    elif model_name == 'simplecnn':
        model = SimpleCNN(num_classes=config.get('num_classes', 10))
        return model
    
    elif model_name == 'quadratic':
        model = QuadraticModel(input_dim=config.get('dimension', 10))
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: Device to train on
        
    Returns:
        train_loss, train_acc: Average training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / total
    train_acc = 100. * correct / total
    
    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): PyTorch model
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        val_loss, val_acc: Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train_model(model, data_loaders, optimizer, criterion, config, device):
    """
    Train a model with the given optimizer and data.
    
    Args:
        model (nn.Module): PyTorch model
        data_loaders (dict): Dictionary containing data loaders
        optimizer: PyTorch optimizer
        criterion: Loss function
        config (dict): Configuration dictionary
        device: Device to train on
        
    Returns:
        history: Dictionary containing training history
    """
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    
    num_epochs = config.get('num_epochs', 10)
    save_dir = config.get('save_dir', './models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
              f'LR: {current_lr:.6f}')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, f'{config["model_name"]}_{config["optimizer"]}_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    # Save final model
    model_path = os.path.join(save_dir, f'{config["model_name"]}_{config["optimizer"]}_final.pth')
    torch.save(model.state_dict(), model_path)
    
    return history


def train_quadratic(model, x_data, y_data, optimizer, criterion, config, device):
    """
    Train a model on the quadratic function data.
    
    Args:
        model (nn.Module): PyTorch model
        x_data (Tensor): Input data
        y_data (Tensor): Target data
        optimizer: PyTorch optimizer
        criterion: Loss function
        config (dict): Configuration dictionary
        device: Device to train on
        
    Returns:
        history: Dictionary containing training history
    """
    num_iterations = config.get('num_iterations', 100)
    
    # Move data to device
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    
    # Initialize history dictionary
    history = {
        'loss': [],
        'params': [],
        'gradients': [],
        'curvature': []
    }
    
    # Initialize previous gradient for curvature estimation
    prev_grad = None
    
    for i in range(num_iterations):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_data)
        loss = criterion(outputs, y_data.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        
        # Store parameters and gradients before update
        params = [p.clone().detach().cpu() for p in model.parameters()]
        grads = [p.grad.clone().detach().cpu() for p in model.parameters()]
        
        # Compute curvature estimate based on gradient change
        if prev_grad is not None:
            grad_diff = [g - pg for g, pg in zip(grads, prev_grad)]
            curvature = [torch.abs(gd).mean().item() for gd in grad_diff]
        else:
            curvature = [0.0] * len(grads)
        
        # Store current gradient for next iteration
        prev_grad = grads
        
        # Optimize
        optimizer.step()
        
        # Print statistics
        if (i + 1) % 10 == 0:
            print(f'Iteration {i+1}/{num_iterations} | Loss: {loss.item():.6f}')
        
        # Update history
        history['loss'].append(loss.item())
        history['params'].append(params)
        history['gradients'].append(grads)
        history['curvature'].append(curvature)
    
    return history


def run_experiment(config):
    """
    Run an experiment based on the configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        results: Dictionary containing experiment results
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data
    from src.preprocess import prepare_data
    data = prepare_data(config)
    
    experiment_type = config.get('experiment_type', 'cifar10')
    model_name = config.get('model_name', 'resnet18')
    optimizer_names = config.get('optimizers', ['sgd', 'adam', 'adabelief', 'acm'])
    
    results = {}
    
    if experiment_type == 'cifar10':
        criterion = nn.CrossEntropyLoss()
        
        for optimizer_name in optimizer_names:
            print(f'\nTraining with optimizer: {optimizer_name}')
            
            # Create model
            model = get_model(model_name, config)
            model = model.to(device)
            
            # Create optimizer
            optimizer = get_optimizer(optimizer_name, model.parameters(), config)
            
            # Train model
            history = train_model(model, data, optimizer, criterion, config, device)
            
            # Store results
            results[optimizer_name] = history
    
    elif experiment_type == 'quadratic':
        criterion = nn.MSELoss()
        
        for optimizer_name in optimizer_names:
            print(f'\nTraining with optimizer: {optimizer_name}')
            
            # Create model
            model = get_model(model_name, config)
            model = model.to(device)
            
            # Create optimizer
            optimizer = get_optimizer(optimizer_name, model.parameters(), config)
            
            # Train model
            history = train_quadratic(model, data['x_data'], data['y_data'], optimizer, criterion, config, device)
            
            # Store results
            results[optimizer_name] = history
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    return results
