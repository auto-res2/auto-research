import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from utils.optimizer import ACMOptimizer

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for CIFAR-10 classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimpleMLP(nn.Module):
    """
    Simple MLP architecture for MNIST classification.
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_synthetic_optimization(synthetic_data, config=None):
    """
    Train on synthetic optimization problems.
    
    Args:
        synthetic_data (dict): Dictionary containing synthetic optimization problems
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Dictionary containing training results
    """
    print("Starting synthetic optimization experiments...")
    
    # Default configuration
    num_iters = 100 if config is None or 'num_iters' not in config else config['num_iters']
    quick_test = False if config is None or 'quick_test' not in config else config['quick_test']
    
    if quick_test:
        print("Running in quick test mode with reduced iterations")
        num_iters = 10
    
    # Extract synthetic data
    quadratic_data = synthetic_data['quadratic']
    A = quadratic_data['A']
    b = quadratic_data['b']
    
    # Define optimizers to compare
    optimizers_dict = {
        "ACM": lambda params, lr: ACMOptimizer(params, lr=lr, beta=0.9, curvature_influence=0.05),
        "Adam": lambda params, lr: optim.Adam(params, lr=lr),
        "SGD_mom": lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9)
    }
    
    # Quadratic function optimization
    print("\n=== Synthetic Experiment: Quadratic Function Optimization ===")
    quadratic_results = {}
    
    # Define the quadratic loss function
    def quadratic_loss(x, A, b):
        return 0.5 * torch.matmul(torch.matmul(x, A), x) - torch.matmul(b, x)
    
    # Run optimization for each optimizer
    for name, opt_class in optimizers_dict.items():
        print(f"\nRunning optimization with {name}")
        
        # Initialize parameters
        x_data = torch.randn(2, requires_grad=True)
        
        # Instantiate the optimizer
        if name == "ACM":
            optimizer = opt_class([x_data], lr=0.1)
        else:
            optimizer = opt_class([x_data], lr=0.1)
        
        # Store losses
        losses = []
        
        # Training loop
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = quadratic_loss(x_data, A, b)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if (i + 1) % (num_iters // 5) == 0 or i == 0:
                print(f"Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
        
        quadratic_results[name] = {
            'losses': losses,
            'final_x': x_data.detach().numpy(),
            'final_loss': losses[-1]
        }
    
    # Rosenbrock function optimization
    print("\n=== Synthetic Experiment: Rosenbrock Function Optimization ===")
    rosenbrock_results = {}
    
    # Define the Rosenbrock loss function
    def rosenbrock_loss(x):
        a = 1.0
        b = 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    # Run optimization for each optimizer
    for name, opt_class in optimizers_dict.items():
        print(f"\nRunning optimization with {name}")
        
        # Initialize parameters
        x_data = torch.tensor([0.0, 0.0], requires_grad=True)
        
        # Instantiate the optimizer
        if name == "ACM":
            optimizer = opt_class([x_data], lr=0.01)
        else:
            optimizer = opt_class([x_data], lr=0.01)
        
        # Store losses
        losses = []
        
        # Training loop
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = rosenbrock_loss(x_data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if (i + 1) % (num_iters // 5) == 0 or i == 0:
                print(f"Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
        
        rosenbrock_results[name] = {
            'losses': losses,
            'final_x': x_data.detach().numpy(),
            'final_loss': losses[-1]
        }
    
    # Save results
    os.makedirs('models/synthetic', exist_ok=True)
    try:
        torch.save({
            'quadratic': quadratic_results,
            'rosenbrock': rosenbrock_results
        }, 'models/synthetic/optimization_results.pt')
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
    
    # Return results
    return {
        'quadratic': quadratic_results,
        'rosenbrock': rosenbrock_results
    }

def train_cifar10(train_loader, test_loader, config=None):
    """
    Train a CNN model on CIFAR-10 dataset.
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Dictionary containing training results
    """
    print("\n=== Deep Neural Network Training on CIFAR-10 ===")
    
    # Default configuration
    num_epochs = 5 if config is None or 'num_epochs' not in config else config['num_epochs']
    quick_test = False if config is None or 'quick_test' not in config else config['quick_test']
    
    if quick_test:
        print("Running in quick test mode with reduced epochs")
        num_epochs = 1
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define optimizers to compare
    optimizers_dict = {
        "ACM": lambda params: ACMOptimizer(params, lr=0.01, beta=0.9, curvature_influence=0.1),
        "Adam": lambda params: optim.Adam(params, lr=0.001),
        "SGD_mom": lambda params: optim.SGD(params, lr=0.01, momentum=0.9)
    }
    
    results = {}
    
    # Train with each optimizer
    for name, opt_class in optimizers_dict.items():
        print(f"\nTraining CIFAR-10 with {name} optimizer")
        
        # Initialize model
        model = SimpleCNN().to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = opt_class(model.parameters())
        
        # Training metrics
        train_losses = []
        test_accuracies = []
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    train_losses.append(running_loss / 100)
                    running_loss = 0.0
                
                # For quick test, break after a few iterations
                if quick_test and i >= 10:
                    break
            
            # Evaluate on test set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # For quick test, break after a few iterations
                    if quick_test and total >= 1000:
                        break
            
            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
            print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy:.2f}%")
        
        training_time = time.time() - start_time
        print(f"Finished training with {name}. Time taken: {training_time:.2f}s")
        
        # Save model
        os.makedirs('models/cifar10', exist_ok=True)
        torch.save(model.state_dict(), f'models/cifar10/model_{name}.pt')
        
        # Store results
        results[name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'training_time': training_time,
            'final_accuracy': test_accuracies[-1]
        }
    
    # Save results
    try:
        torch.save(results, 'models/cifar10/training_results.pt')
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
    
    return results

def train_mnist_ablation(train_loader, test_loader, config=None):
    """
    Perform ablation study on MNIST dataset with different hyperparameters.
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Dictionary containing ablation study results
    """
    print("\n=== Ablation Study & Hyperparameter Sensitivity Analysis on MNIST ===")
    
    # Default configuration
    num_epochs = 3 if config is None or 'num_epochs' not in config else config['num_epochs']
    quick_test = False if config is None or 'quick_test' not in config else config['quick_test']
    
    if quick_test:
        print("Running in quick test mode with reduced epochs and hyperparameters")
        num_epochs = 1
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameter configurations for ACM
    if quick_test:
        # Reduced set for quick testing
        lr_values = [0.01]
        beta_values = [0.9]
        curvature_values = [0.1]
    else:
        # Full set for complete ablation study
        lr_values = [0.001, 0.01, 0.1]
        beta_values = [0.8, 0.9, 0.95]
        curvature_values = [0.01, 0.1, 0.5]
    
    results = {}
    
    # Iterate through hyperparameter combinations
    for lr in lr_values:
        for beta in beta_values:
            for curvature in curvature_values:
                config_name = f"lr{lr}_beta{beta}_curv{curvature}"
                print(f"\nTraining MNIST with ACM hyperparameters: {config_name}")
                
                # Initialize model
                model = SimpleMLP().to(device)
                
                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = ACMOptimizer(
                    model.parameters(),
                    lr=lr,
                    beta=beta,
                    curvature_influence=curvature
                )
                
                # Training metrics
                train_losses = []
                test_accuracies = []
                
                # Training loop
                start_time = time.time()
                
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0
                    
                    for i, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        # Zero the parameter gradients
                        optimizer.zero_grad()
                        
                        # Forward + backward + optimize
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        # Print statistics
                        running_loss += loss.item()
                        if i % 100 == 99:  # Print every 100 mini-batches
                            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                            train_losses.append(running_loss / 100)
                            running_loss = 0.0
                        
                        # For quick test, break after a few iterations
                        if quick_test and i >= 10:
                            break
                    
                    # Evaluate on test set
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            
                            # For quick test, break after a few iterations
                            if quick_test and total >= 1000:
                                break
                    
                    accuracy = 100 * correct / total
                    test_accuracies.append(accuracy)
                    print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy:.2f}%")
                
                training_time = time.time() - start_time
                print(f"Finished training with config {config_name}. Time taken: {training_time:.2f}s")
                
                # Save model
                os.makedirs('models/mnist', exist_ok=True)
                torch.save(model.state_dict(), f'models/mnist/model_{config_name}.pt')
                
                # Store results
                results[config_name] = {
                    'lr': lr,
                    'beta': beta,
                    'curvature_influence': curvature,
                    'train_losses': train_losses,
                    'test_accuracies': test_accuracies,
                    'training_time': training_time,
                    'final_accuracy': test_accuracies[-1]
                }
    
    # Save results
    try:
        torch.save(results, 'models/mnist/ablation_results.pt')
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
    
    return results

def train_models(data, config=None):
    """
    Main function to train all models.
    
    Args:
        data (dict): Dictionary containing all preprocessed data
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Dictionary containing all training results
    """
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Train on synthetic optimization problems
    synthetic_results = train_synthetic_optimization(data['synthetic'], config)
    
    # Train on CIFAR-10
    cifar10_results = train_cifar10(data['cifar10'][0], data['cifar10'][1], config)
    
    # Train on MNIST with ablation study
    mnist_results = train_mnist_ablation(data['mnist'][0], data['mnist'][1], config)
    
    # Return all results
    return {
        'synthetic': synthetic_results,
        'cifar10': cifar10_results,
        'mnist': mnist_results
    }

if __name__ == "__main__":
    # When run directly, import and preprocess data, then train models
    from preprocess import preprocess_data
    
    # Preprocess data
    data = preprocess_data()
    
    # Train with quick test mode
    config = {'quick_test': True}
    train_models(data, config)
    
    print("Model training completed successfully.")
