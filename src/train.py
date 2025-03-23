import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from src.optimizers import ACMOptimizer

class SimpleCNN(nn.Module):
    """
    A simple CNN for CIFAR-10 classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MNISTNet(nn.Module):
    """
    A simple neural network for MNIST classification.
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, test_loader, optimizer_name, optimizer_kwargs, 
               device, epochs=10, log_interval=100, save_path=None):
    """
    Train a model using the specified optimizer and evaluate on test data.
    
    Args:
        model: neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer_name: name of optimizer to use (ACM, Adam, SGD_mom)
        optimizer_kwargs: keyword arguments for the optimizer
        device: device to train on (cuda or cpu)
        epochs: number of training epochs
        log_interval: how often to log training progress
        save_path: where to save the trained model
        
    Returns:
        training_losses: list of training losses per epoch
        test_accuracies: list of test accuracies per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer
    if optimizer_name == 'ACM':
        optimizer = ACMOptimizer(model.parameters(), **optimizer_kwargs)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    elif optimizer_name == 'SGD_mom':
        optimizer = optim.SGD(model.parameters(), momentum=0.9, **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    training_losses = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_loss)
        print(f'Epoch {epoch}: Average training loss: {avg_loss:.4f}')
        
        # Testing
        accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(accuracy)
        print(f'Epoch {epoch}: Test accuracy: {accuracy:.4f}')
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        
    return training_losses, test_accuracies

def run_synthetic_optimization(optimizer_name, optimizer_kwargs, num_iters=100, seed=0):
    """
    Run optimization on synthetic functions (quadratic and Rosenbrock).
    
    Args:
        optimizer_name: name of optimizer to use (ACM, Adam, SGD_mom)
        optimizer_kwargs: keyword arguments for the optimizer
        num_iters: number of optimization iterations
        seed: random seed for reproducibility
        
    Returns:
        results: dictionary with optimization losses for both functions
    """
    torch.manual_seed(seed)
    
    # Define quadratic function parameters
    A = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
    b = torch.tensor([1.0, 1.0])
    
    # Create initial points
    x_quad = torch.randn(2, requires_grad=True)
    x_rosen = torch.tensor([0.0, 0.0], requires_grad=True)
    
    # Select optimizer for quadratic function
    if optimizer_name == 'ACM':
        quad_optimizer = ACMOptimizer([x_quad], **optimizer_kwargs)
        rosen_optimizer = ACMOptimizer([x_rosen], **optimizer_kwargs)
    elif optimizer_name == 'Adam':
        quad_optimizer = optim.Adam([x_quad], **optimizer_kwargs)
        rosen_optimizer = optim.Adam([x_rosen], **optimizer_kwargs)
    elif optimizer_name == 'SGD_mom':
        quad_optimizer = optim.SGD([x_quad], momentum=0.9, **optimizer_kwargs)
        rosen_optimizer = optim.SGD([x_rosen], momentum=0.9, **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Optimization results
    quadratic_losses = []
    rosenbrock_losses = []
    
    # Quadratic function optimization
    for i in range(num_iters):
        quad_optimizer.zero_grad()
        quad_loss = 0.5 * x_quad @ A @ x_quad - b @ x_quad
        quad_loss.backward()
        quad_optimizer.step()
        quadratic_losses.append(quad_loss.item())
        
    # Rosenbrock function optimization
    for i in range(num_iters):
        rosen_optimizer.zero_grad()
        a, b_coef = 1.0, 100.0
        rosen_loss = (a - x_rosen[0])**2 + b_coef * (x_rosen[1] - x_rosen[0]**2)**2
        rosen_loss.backward()
        rosen_optimizer.step()
        rosenbrock_losses.append(rosen_loss.item())
    
    results = {
        'quadratic': quadratic_losses,
        'rosenbrock': rosenbrock_losses,
        'final_x_quad': x_quad.detach().numpy(),
        'final_x_rosen': x_rosen.detach().numpy()
    }
    
    return results
