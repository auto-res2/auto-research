"""
Model training module for the auto-research project.
Contains functions for training models for the experiments.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from src.utils.optimizers import ACM
from src.preprocess import set_seed, get_device, synthetic_function

class SimpleNet(nn.Module):
    """
    Simple neural network for the two-moons classification task.
    
    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
    """
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train_cifar10_model(trainloader, testloader, optimizer_name='acm', 
                        num_epochs=3, lr=0.001, beta=0.9, curvature_scale=1.0,
                        log_dir='./logs', model_dir='./models'):
    """
    Train a ResNet-18 model on the CIFAR-10 dataset.
    
    Args:
        trainloader (DataLoader): Training data loader
        testloader (DataLoader): Test data loader
        optimizer_name (str): Name of the optimizer to use ('acm' or 'adam')
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        beta (float): Momentum coefficient for ACM
        curvature_scale (float): Curvature scale for ACM
        log_dir (str): Directory for TensorBoard logs
        model_dir (str): Directory for saving models
        
    Returns:
        tuple: (model, train_losses, test_accuracies) containing the trained model,
               training losses, and test accuracies
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"{log_dir}/cifar10_{optimizer_name}")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize model
    model = models.resnet18(pretrained=False, num_classes=10)
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    if optimizer_name.lower() == 'acm':
        optimizer = ACM(model.parameters(), lr=lr, beta=beta, curvature_scale=curvature_scale)
        print(f"Using ACM optimizer with lr={lr}, beta={beta}, curvature_scale={curvature_scale}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f"Using Adam optimizer with lr={lr}")
    
    # Training loop
    train_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Calculate gradient norm for logging
            grad_norm = sum(p.grad.data.norm().item() for p in model.parameters() if p.grad is not None)
            
            # Log to TensorBoard
            global_step = epoch * len(trainloader) + i
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Gradient_norm', grad_norm, global_step)
            
            # Print progress
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], "
                      f"Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate accuracy
        accuracy = 100.0 * correct / total
        test_accuracies.append(accuracy)
        
        # Log to TensorBoard
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"{model_dir}/cifar10_{optimizer_name}_best.pth")
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, "
              f"Test Accuracy: {accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), f"{model_dir}/cifar10_{optimizer_name}_final.pth")
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Close TensorBoard writer
    writer.close()
    
    return model, train_losses, test_accuracies

def run_synthetic_experiment(optimizer_class, optimizer_kwargs, num_steps=50):
    """
    Run the synthetic function optimization experiment.
    
    Args:
        optimizer_class: Optimizer class to use
        optimizer_kwargs (dict): Keyword arguments for the optimizer
        num_steps (int): Number of optimization steps
        
    Returns:
        numpy.ndarray: Trajectory of the optimization process
    """
    # Starting point in R^2
    xy = torch.tensor([4.0, 4.0], requires_grad=True)
    trajectory = [xy.detach().clone().numpy()]
    
    # Initialize optimizer
    optimizer = optimizer_class([xy], **optimizer_kwargs)
    
    # Optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = synthetic_function(xy)
        loss.backward()
        optimizer.step()
        trajectory.append(xy.detach().clone().numpy())
        
        if (step + 1) % 10 == 0:
            print(f"Step [{step+1}/{num_steps}], Loss: {loss.item():.4f}, "
                  f"Position: [{xy[0].item():.4f}, {xy[1].item():.4f}]")
    
    return np.array(trajectory)

def train_two_moons_model(X_train, y_train, X_val, y_val, optimizer_name='acm',
                         num_epochs=50, lr=0.05, beta=0.9, curvature_scale=1.0,
                         hidden_dim=10, log_dir='./logs', model_dir='./models'):
    """
    Train a simple neural network on the two-moons dataset.
    
    Args:
        X_train (torch.Tensor): Training features
        y_train (torch.Tensor): Training labels
        X_val (torch.Tensor): Validation features
        y_val (torch.Tensor): Validation labels
        optimizer_name (str): Name of the optimizer to use ('acm' or 'adam')
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        beta (float): Momentum coefficient for ACM
        curvature_scale (float): Curvature scale for ACM
        hidden_dim (int): Hidden layer dimension
        log_dir (str): Directory for TensorBoard logs
        model_dir (str): Directory for saving models
        
    Returns:
        tuple: (model, losses, accuracy) containing the trained model,
               training losses, and final validation accuracy
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"{log_dir}/two_moons_{optimizer_name}_b{beta}_cs{curvature_scale}")
    
    # Get device
    device = get_device()
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    # Initialize model
    model = SimpleNet(input_dim=2, hidden_dim=hidden_dim, output_dim=2)
    model = model.to(device)
    
    # Initialize optimizer
    if optimizer_name.lower() == 'acm':
        optimizer = ACM(model.parameters(), lr=lr, beta=beta, curvature_scale=curvature_scale)
        print(f"Using ACM optimizer with lr={lr}, beta={beta}, curvature_scale={curvature_scale}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f"Using Adam optimizer with lr={lr}")
    
    # Training loop
    losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = F.cross_entropy(outputs, y_train)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        losses.append(loss.item())
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == y_val).float().mean().item()
    
    # Log final accuracy
    writer.add_scalar('Accuracy/val', accuracy, 0)
    
    # Save model
    torch.save(model.state_dict(), 
               f"{model_dir}/two_moons_{optimizer_name}_b{beta}_cs{curvature_scale}.pth")
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final validation accuracy: {accuracy:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    
    return model, losses, accuracy
