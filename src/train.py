#!/usr/bin/env python3
"""
Training module for ACM optimizer experiments.

This module contains functions for training models using different optimizers,
including the ACM optimizer, and comparing their performance.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from src.utils.optimizers import ACM


class SimpleCNN(nn.Module):
    """
    Simple CNN model for CIFAR-10 classification.
    
    This is a basic convolutional neural network with two convolutional layers
    followed by two fully connected layers.
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize the SimpleCNN model.
        
        Args:
            num_classes (int): Number of output classes.
        """
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name, num_classes=10, device=None):
    """
    Get a model for training.
    
    Args:
        model_name (str): Name of the model to use.
        num_classes (int): Number of output classes.
        device (torch.device): Device to use for training.
        
    Returns:
        torch.nn.Module: Model for training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name.lower() == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes).to(device)
    elif model_name.lower() == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def get_optimizer(optimizer_name, model_params, config, experiment_key):
    """
    Get an optimizer for training.
    
    Args:
        optimizer_name (str): Name of the optimizer to use.
        model_params (iterable): Model parameters to optimize.
        config (dict): Configuration dictionary.
        experiment_key (str): Key for the experiment in the config.
        
    Returns:
        torch.optim.Optimizer: Optimizer for training.
    """
    optimizer_config = config[experiment_key]['optimizers'][optimizer_name.lower()]
    
    # Ensure all parameters are of the correct type
    lr = float(optimizer_config['lr'])
    weight_decay = float(optimizer_config.get('weight_decay', 0))
    
    if optimizer_name.lower() == 'acm':
        beta = float(optimizer_config['beta'])
        optimizer = ACM(
            model_params,
            lr=lr,
            beta=beta,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        momentum = float(optimizer_config.get('momentum', 0))
        optimizer = optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adam':
        betas = tuple(float(b) for b in optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.Adam(
            model_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def train_and_evaluate(model, optimizer, train_loader, test_loader, criterion, 
                      num_epochs, device, optimizer_name="Optimizer"):
    """
    Train a model and evaluate its performance.
    
    Args:
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        criterion (torch.nn.Module): Loss function.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): Device to use for training.
        optimizer_name (str): Name of the optimizer for logging.
        
    Returns:
        tuple: (train_losses, test_accuracies) for each epoch.
    """
    train_losses = []
    test_accuracies = []
    
    # Ensure we're using Python lists, not range objects
    if isinstance(num_epochs, range):
        num_epochs = list(num_epochs)
    
    print(f"--- Training with {optimizer_name} ---")
    
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
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Update statistics
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate accuracy
        test_acc = correct / total
        test_accuracies.append(test_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Test Acc: {test_acc:.4f}")
    
    # Convert to lists to ensure compatibility
    return list(train_losses), list(test_accuracies)


def train_synthetic_function(optimizer_type, synthetic_loss_fn, num_steps=100, lr=0.1, device=None):
    """
    Optimize a synthetic function using a given optimizer.
    
    Args:
        optimizer_type (str): Type of optimizer to use ('ACM', 'SGD', or 'Adam').
        synthetic_loss_fn (callable): Synthetic loss function to optimize.
        num_steps (int): Number of optimization steps.
        lr (float): Learning rate.
        device (torch.device): Device to use for optimization.
        
    Returns:
        numpy.ndarray: Trajectory of parameter values during optimization.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize parameter vector (requires gradient)
    params = torch.tensor([1.5, 1.5], requires_grad=True, device=device)
    trajectory = [params.detach().cpu().numpy().copy()]
    
    # Initialize optimizer
    if optimizer_type.lower() == 'acm':
        opt = ACM([params], lr=lr, beta=0.9)
    elif optimizer_type.lower() == 'sgd':
        opt = optim.SGD([params], lr=lr)
    elif optimizer_type.lower() == 'adam':
        opt = optim.Adam([params], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Optimization loop
    for i in range(num_steps):
        opt.zero_grad()
        loss = synthetic_loss_fn(params)
        loss.backward()
        opt.step()
        trajectory.append(params.detach().cpu().numpy().copy())
    
    return np.array(trajectory)


def plot_training_results(epochs, losses_acm, losses_sgd, losses_adam, 
                         acc_acm, acc_sgd, acc_adam, save_path=None):
    """
    Plot training loss and test accuracy curves for different optimizers.
    
    Args:
        epochs (list): List of epoch numbers.
        losses_acm (list): Training losses for ACM optimizer.
        losses_sgd (list): Training losses for SGD optimizer.
        losses_adam (list): Training losses for Adam optimizer.
        acc_acm (list): Test accuracies for ACM optimizer.
        acc_sgd (list): Test accuracies for SGD optimizer.
        acc_adam (list): Test accuracies for Adam optimizer.
        save_path (str, optional): Path to save the plot.
    """
    # Convert all inputs to lists to ensure compatibility
    epochs = list(epochs)
    losses_acm = list(losses_acm)
    losses_sgd = list(losses_sgd)
    losses_adam = list(losses_adam)
    acc_acm = list(acc_acm)
    acc_sgd = list(acc_sgd)
    acc_adam = list(acc_adam)
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses_acm, marker='o', label='ACM')
    plt.plot(epochs, losses_sgd, marker='o', label='SGD')
    plt.plot(epochs, losses_adam, marker='o', label='Adam')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_acm, marker='s', label='ACM')
    plt.plot(epochs, acc_sgd, marker='s', label='SGD')
    plt.plot(epochs, acc_adam, marker='s', label='Adam')
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.suptitle("Experiment 1 Results: Convergence Speed and Generalization")
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_optimization_trajectories(traj_acm, traj_sgd, traj_adam, 
                                  synthetic_loss_fn, save_path=None):
    """
    Plot optimization trajectories for different optimizers on a synthetic loss landscape.
    
    Args:
        traj_acm (numpy.ndarray): Trajectory for ACM optimizer.
        traj_sgd (numpy.ndarray): Trajectory for SGD optimizer.
        traj_adam (numpy.ndarray): Trajectory for Adam optimizer.
        synthetic_loss_fn (callable): Synthetic loss function.
        save_path (str, optional): Path to save the plot.
    """
    # Create meshgrid for contour plot
    x_vals = np.linspace(-1.5, 1.5, 100)
    y_vals = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute loss values for the meshgrid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            params = torch.tensor([X[i, j], Y[i, j]], requires_grad=False)
            Z[i, j] = synthetic_loss_fn(params).item()
    
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.plot(traj_acm[:, 0], traj_acm[:, 1], marker='o', label='ACM')
    plt.plot(traj_sgd[:, 0], traj_sgd[:, 1], marker='x', label='SGD')
    plt.plot(traj_adam[:, 0], traj_adam[:, 1], marker='s', label='Adam')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Trajectories on Synthetic Loss Landscape')
    plt.legend()
    
    # Save the plot if a save path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()


def save_model(model, optimizer, epoch, loss, accuracy, save_path):
    """
    Save a trained model and optimizer state.
    
    Args:
        model (torch.nn.Module): Trained model.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        epoch (int): Current epoch.
        loss (float): Current loss.
        accuracy (float): Current accuracy.
        save_path (str): Path to save the model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, save_path)
    
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Load configuration
    with open('config/acm_experiments.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test model creation
    model = get_model('simple_cnn', num_classes=10, device=device)
    print(f"Model created: {model.__class__.__name__}")
    
    # Test optimizer creation
    optimizer = get_optimizer('acm', model.parameters(), config, 'experiment1')
    print(f"Optimizer created: {optimizer.__class__.__name__}")
    
    # Test synthetic function optimization
    def synthetic_loss(params):
        x, y = params[0], params[1]
        return 50.0 * x**2 + 1.0 * y**2
    
    trajectory = train_synthetic_function('acm', synthetic_loss, num_steps=10, device=device)
    print(f"Synthetic optimization trajectory shape: {trajectory.shape}")
