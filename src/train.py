"""
Model training module for ACM optimizer experiments.

This module implements training functions for:
1. ResNet-18 on CIFAR-10 (Real-World Convergence Experiment)
2. Optimization on synthetic functions (Synthetic Loss Landscape Experiment)
3. Simple CNN with hyperparameter grid search (Hyperparameter Sensitivity Experiment)
"""

import time
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from src.utils.optimizer import ACM

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    """
    Simple CNN model for CIFAR-10 classification.
    Used in the hyperparameter sensitivity experiment.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_resnet_cifar10(train_loader, test_loader, optimizer_name, optimizer_params, test_run=False):
    """
    Train ResNet-18 on CIFAR-10 with the specified optimizer.
    
    Args:
        train_loader: PyTorch DataLoader for training data
        test_loader: PyTorch DataLoader for test data
        optimizer_name (str): Name of the optimizer ('ACM', 'Adam', or 'SGD')
        optimizer_params (dict): Parameters for the optimizer
        test_run (bool): If True, use fewer epochs for quick testing
    
    Returns:
        tuple: (model, history) - Trained model and training history
    """
    print(f"\n--- Training ResNet-18 with {optimizer_name} optimizer ---")
    
    # Initialize ResNet-18 model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 (10 classes)
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    if optimizer_name == "ACM":
        # Handle weight_decay separately for ACM
        weight_decay = optimizer_params.pop('weight_decay', 0) if isinstance(optimizer_params, dict) else 0
        optimizer = ACM(model.parameters(), weight_decay=weight_decay, **optimizer_params)
        # Restore weight_decay to params dict if it was there
        if weight_decay != 0:
            optimizer_params['weight_decay'] = weight_decay
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Number of epochs
    num_epochs = 1 if test_run else 30
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
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
        
        # Calculate epoch statistics
        epoch_time = time.time() - start_time
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate validation statistics
        val_loss = val_loss / len(test_loader.dataset)
        val_acc = 100.0 * correct / total
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Time: {epoch_time:.2f}s")
    
    return model, history

def optimize_synthetic_function(fn, x_init, optimizer_type, alpha=0.1, beta=0.05, n_iters=50, test_run=False):
    """
    Optimize a synthetic function using the specified optimizer.
    
    Args:
        fn: Function to optimize
        x_init (torch.Tensor): Initial point
        optimizer_type (str): Type of optimizer ('acm' or 'sgd')
        alpha (float): Learning rate
        beta (float): Curvature influence factor (for ACM)
        n_iters (int): Number of iterations
        test_run (bool): If True, use fewer iterations for quick testing
    
    Returns:
        tuple: (trajectory, adaptive_lrs) - Optimization trajectory and adaptive learning rates
    """
    # If in test mode, use fewer iterations
    if test_run:
        n_iters = 10
    
    # Initialize variables
    x = x_init.clone().detach().requires_grad_(True)
    trajectory = [x.detach().numpy().copy()]
    adaptive_lrs = []
    grad_prev = None
    
    # Optimization loop
    for i in range(n_iters):
        # Compute loss and gradients
        loss = fn(x)
        loss.backward()
        
        # Get current gradient
        grad = x.grad.data.clone()
        
        # Estimate curvature
        if grad_prev is None:
            curvature_est = torch.zeros_like(grad)
        else:
            curvature_est = (grad - grad_prev).abs()
        
        # Store current gradient for next iteration
        grad_prev = grad.clone()
        
        # Update parameters based on optimizer type
        if optimizer_type == "acm":
            # Compute adaptive learning rate
            adaptive_lr = alpha / (1.0 + beta * curvature_est)
            adaptive_lrs.append(adaptive_lr.mean().item())
            
            # Update parameters
            with torch.no_grad():
                x.data -= adaptive_lr * grad
        elif optimizer_type == "sgd":
            # Update parameters with constant learning rate
            with torch.no_grad():
                x.data -= alpha * grad
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Reset gradients
        x.grad.zero_()
        
        # Store current position
        trajectory.append(x.detach().numpy().copy())
        
        # Print progress for every 10% of iterations
        if i % max(1, n_iters // 10) == 0 or i == n_iters - 1:
            print(f"Iteration {i+1}/{n_iters}: Loss = {loss.item():.6f}")
    
    return np.array(trajectory), adaptive_lrs

def train_cnn_hyperparameter_search(train_loader, test_loader, optimizer_type, param_grid, test_run=False):
    """
    Train a simple CNN with different hyperparameter settings.
    
    Args:
        train_loader: PyTorch DataLoader for training data
        test_loader: PyTorch DataLoader for test data
        optimizer_type (str): Type of optimizer ('ACM' or 'Adam')
        param_grid (dict): Grid of hyperparameters to search
        test_run (bool): If True, use fewer epochs for quick testing
    
    Returns:
        list: Results for each hyperparameter setting
    """
    # Number of epochs
    num_epochs = 2 if test_run else 20
    
    # Results list
    results = []
    
    # Grid search
    if optimizer_type == "ACM":
        # ACM hyperparameter grid
        for lr in param_grid["lr"]:
            for beta in param_grid["beta"]:
                print(f"\n--- Training CNN with ACM: lr={lr}, beta={beta} ---")
                
                # Initialize model
                model = SimpleCNN().to(device)
                
                # Initialize optimizer
                optimizer = ACM(
                    model.parameters(),
                    lr=lr,
                    beta=beta,
                    weight_decay=0
                )
                
                # Train model
                train_model_with_params(model, optimizer, train_loader, test_loader, num_epochs)
                
                # Evaluate model
                val_loss, val_acc = evaluate_model(model, test_loader)
                
                # Store results
                results.append({
                    "optimizer": "ACM",
                    "lr": lr,
                    "beta": beta,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
                
                print(f"ACM: lr={lr}, beta={beta} -> Val Acc={val_acc:.2f}%, Val Loss={val_loss:.4f}")
    
    elif optimizer_type == "Adam":
        # Adam hyperparameter grid
        for lr in param_grid["lr"]:
            print(f"\n--- Training CNN with Adam: lr={lr} ---")
            
            # Initialize model
            model = SimpleCNN().to(device)
            
            # Initialize optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=1e-4
            )
            
            # Train model
            train_model_with_params(model, optimizer, train_loader, test_loader, num_epochs)
            
            # Evaluate model
            val_loss, val_acc = evaluate_model(model, test_loader)
            
            # Store results
            results.append({
                "optimizer": "Adam",
                "lr": lr,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            
            print(f"Adam: lr={lr} -> Val Acc={val_acc:.2f}%, Val Loss={val_loss:.4f}")
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return results

def train_model_with_params(model, optimizer, train_loader, test_loader, num_epochs):
    """
    Train a model with the given optimizer and parameters.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        train_loader: PyTorch DataLoader for training data
        test_loader: PyTorch DataLoader for test data
        num_epochs (int): Number of epochs to train
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
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
        
        # Calculate epoch statistics
        train_loss = running_loss / len(train_loader.dataset)
        
        # Print progress every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}")

def evaluate_model(model, test_loader):
    """
    Evaluate a model on the test set.
    
    Args:
        model: PyTorch model
        test_loader: PyTorch DataLoader for test data
    
    Returns:
        tuple: (val_loss, val_acc) - Validation loss and accuracy
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate validation statistics
    val_loss = val_loss / len(test_loader.dataset)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc

if __name__ == "__main__":
    # Test model initialization
    model = SimpleCNN().to(device)
    print(f"SimpleCNN model created successfully.")
    print(model)
