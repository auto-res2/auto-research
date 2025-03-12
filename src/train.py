import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.optimizers import ACMOptimizer
from src.utils.models import SimpleCNN, MNISTNet

def train_synthetic(optimizer_name, num_iters=100, quick_test=False):
    """
    Train on synthetic optimization problems
    """
    print(f"=== Synthetic Experiment: Quadratic Function Optimization with {optimizer_name} ===")
    torch.manual_seed(0)
    
    # Example 2D quadratic. A is positive definite.
    A = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
    b = torch.tensor([1.0, 1.0])
    
    # Define quadratic loss function
    def quadratic_loss(x, A, b):
        return 0.5 * x @ A @ x - b @ x
    
    # Initialize parameters
    x_data = torch.randn(2, requires_grad=True)
    
    # Instantiate the optimizer
    if optimizer_name == "ACM":
        optimizer = ACMOptimizer([x_data], lr=0.1, beta=0.9, curvature_influence=0.05)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam([x_data], lr=0.1)
    elif optimizer_name == "SGD_mom":
        optimizer = optim.SGD([x_data], lr=0.1, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Limit iterations for quick test
    if quick_test:
        num_iters = 10
    
    losses = []
    
    # Training loop
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = quadratic_loss(x_data, A, b)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (i + 1) % (max(1, num_iters // 5)) == 0 or i == 0:
            print(f"Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
    
    return losses, x_data.detach().numpy()

def train_rosenbrock(optimizer_name, num_iters=100, quick_test=False):
    """
    Train on Rosenbrock function
    """
    print(f"=== Synthetic Experiment: Rosenbrock Function Optimization with {optimizer_name} ===")
    torch.manual_seed(0)
    
    # Define Rosenbrock function
    def rosenbrock_loss(x):
        a = 1.0
        b = 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    # Initialize parameters
    x_data = torch.tensor([0.0, 0.0], requires_grad=True)
    
    # Instantiate the optimizer
    if optimizer_name == "ACM":
        optimizer = ACMOptimizer([x_data], lr=0.01, beta=0.9, curvature_influence=0.05)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam([x_data], lr=0.01)
    elif optimizer_name == "SGD_mom":
        optimizer = optim.SGD([x_data], lr=0.01, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Limit iterations for quick test
    if quick_test:
        num_iters = 10
    
    losses = []
    
    # Training loop
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = rosenbrock_loss(x_data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (i + 1) % (max(1, num_iters // 5)) == 0 or i == 0:
            print(f"Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
    
    return losses, x_data.detach().numpy()

def train_cifar10(optimizer_name, trainloader, testloader, epochs=10, quick_test=False):
    """
    Train a CNN on CIFAR-10
    """
    print(f"=== CIFAR-10 Training with {optimizer_name} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate the optimizer
    if optimizer_name == "ACM":
        optimizer = ACMOptimizer(model.parameters(), lr=0.01, beta=0.9, curvature_influence=0.05)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == "SGD_mom":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Limit epochs and data for quick test
    if quick_test:
        epochs = 1
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainloader.dataset, range(100)),
            batch_size=trainloader.batch_size,
            shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(testloader.dataset, range(100)),
            batch_size=testloader.batch_size,
            shuffle=False
        )
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test the model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        test_accs.append(test_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    # Save model
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), f'./models/cifar10_{optimizer_name}.pth')
    
    return train_losses, train_accs, test_accs

def train_mnist_ablation(trainloader, testloader, lr_values, beta_values, curvature_values, epochs=5, quick_test=False):
    """
    Ablation study on MNIST with different hyperparameters for ACM
    """
    print("=== MNIST Ablation Study for ACM Optimizer ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Limit epochs and data for quick test
    if quick_test:
        epochs = 1
        lr_values = [lr_values[0]]
        beta_values = [beta_values[0]]
        curvature_values = [curvature_values[0]]
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainloader.dataset, range(100)),
            batch_size=trainloader.batch_size,
            shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(testloader.dataset, range(100)),
            batch_size=testloader.batch_size,
            shuffle=False
        )
    
    results = {}
    
    for lr in lr_values:
        for beta in beta_values:
            for curvature in curvature_values:
                config_name = f"lr={lr}_beta={beta}_curv={curvature}"
                print(f"\nTraining with {config_name}")
                
                # Create model
                model = MNISTNet().to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = ACMOptimizer(model.parameters(), lr=lr, beta=beta, curvature_influence=curvature)
                
                train_losses = []
                test_accs = []
                
                # Training loop
                for epoch in range(epochs):
                    model.train()
                    running_loss = 0.0
                    
                    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")):
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                    
                    train_loss = running_loss / len(trainloader)
                    train_losses.append(train_loss)
                    
                    # Test the model
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, targets in testloader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = model(inputs)
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                    
                    test_acc = 100. * correct / total
                    test_accs.append(test_acc)
                    
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%")
                
                # Save model
                os.makedirs('./models', exist_ok=True)
                torch.save(model.state_dict(), f'./models/mnist_acm_{config_name}.pth')
                
                results[config_name] = {
                    'train_losses': train_losses,
                    'test_accs': test_accs,
                    'final_test_acc': test_accs[-1]
                }
    
    return results
