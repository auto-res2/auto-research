import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18
import time
import os
from utils.optimizer import ACMOptimizer

def train_cifar10(train_loader, test_loader, optimizer_name, num_epochs=5, 
                lr=0.1, beta=0.9, save_model=True, device=None):
    """
    Train ResNet-18 on CIFAR-10 dataset
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer_name (str): Name of optimizer to use ('acm', 'adam', or 'sgd')
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        beta (float): Beta parameter for ACM optimizer
        save_model (bool): Whether to save the trained model
        device: Device to use for training (None for auto-detection)
        
    Returns:
        dict: Dictionary containing training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = resnet18(num_classes=10)
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if optimizer_name.lower() == 'acm':
        optimizer = ACMOptimizer(model.parameters(), lr=lr, beta=beta)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Training history
    history = {
        'train_loss': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
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
            
            running_loss += loss.item()
        
        # Calculate average loss
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate accuracy
        accuracy = correct / total
        history['test_acc'].append(accuracy)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {avg_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    # Save model if requested
    if save_model:
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 
                  f'models/cifar10_resnet18_{optimizer_name}.pth')
    
    return history

def optimize_rosenbrock(optimizer_name, num_iters=500, lr=1e-3, beta=0.9):
    """
    Optimize the Rosenbrock function
    
    Args:
        optimizer_name (str): Name of optimizer to use ('acm', 'adam', or 'sgd')
        num_iters (int): Number of iterations
        lr (float): Learning rate
        beta (float): Beta parameter for ACM optimizer
        
    Returns:
        tuple: (trajectory, loss_values)
    """
    # Define Rosenbrock function
    def rosenbrock(x):
        a = 1.0
        b = 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    # Initialize parameters at (-1.5, 1.5)
    x = torch.tensor([-1.5, 1.5], requires_grad=True)
    
    # Create optimizer
    if optimizer_name.lower() == 'acm':
        optimizer = ACMOptimizer([x], lr=lr, beta=beta)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam([x], lr=lr)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD([x], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Optimization loop
    trajectory = []
    loss_values = []
    
    for i in range(num_iters):
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute loss
        loss = rosenbrock(x)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Record trajectory and loss
        trajectory.append(x.detach().clone().numpy())
        loss_values.append(loss.item())
        
        # Print progress every 100 iterations
        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}/{num_iters} - Loss: {loss.item():.4f}")
    
    return np.array(trajectory), loss_values

def train_text_classifier(train_loader, test_loader, vocab, optimizer_name, 
                        num_epochs=5, lr=1e-3, beta=0.9, device=None):
    """
    Train a simple text classifier on AG_NEWS dataset
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        vocab: Vocabulary object
        optimizer_name (str): Name of optimizer to use ('acm', 'adam', or 'sgd')
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        beta (float): Beta parameter for ACM optimizer
        device: Device to use for training (None for auto-detection)
        
    Returns:
        dict: Dictionary containing training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model
    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_class):
            super(TextClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.fc = nn.Linear(embed_dim, num_class)
        
        def forward(self, text):
            embedded = self.embedding(text)
            # Average word embeddings
            pooled = embedded.mean(dim=1)
            return self.fc(pooled)
    
    # Create model
    vocab_size = len(vocab)
    embed_dim = 64
    num_class = 4  # AG_NEWS has 4 classes
    
    model = TextClassifier(vocab_size, embed_dim, num_class).to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if optimizer_name.lower() == 'acm':
        optimizer = ACMOptimizer(model.parameters(), lr=lr, beta=beta)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Training history
    history = {
        'train_loss': [],
        'test_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for labels, texts in train_loader:
            labels, texts = labels.to(device), texts.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for labels, texts in test_loader:
                labels, texts = labels.to(device), texts.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        accuracy = correct / total
        history['test_acc'].append(accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {avg_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}")
    
    return history
