import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.optimizer import ACMOptimizer

class SimpleCNN(nn.Module):
    """
    Simple CNN model for CIFAR-10 classification.
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

class MNISTNet(nn.Module):
    """
    Simple network for MNIST classification.
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
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_synthetic_function(optimizer_name, synthetic_data, config):
    """
    Train on synthetic optimization problems.
    
    Args:
        optimizer_name (str): Name of the optimizer to use
        synthetic_data (dict): Dictionary containing synthetic data
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing training results
    """
    # Extract parameters from config
    num_iters = config.get('synthetic_num_iters', 100)
    quick_test = config.get('quick_test', False)
    
    if quick_test:
        num_iters = min(num_iters, 10)  # Limit iterations for quick testing
    
    # Quadratic function optimization
    A = synthetic_data['quadratic']['A']
    b = synthetic_data['quadratic']['b']
    
    # Initialize parameters
    x = torch.randn(2, requires_grad=True)
    
    # Define the quadratic loss function
    def quadratic_loss(x, A, b):
        return 0.5 * x @ A @ x - b @ x
    
    # Initialize optimizer based on name
    if optimizer_name == 'ACM':
        optimizer = ACMOptimizer([x], lr=0.1, beta=0.9, curvature_influence=0.05)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam([x], lr=0.1)
    elif optimizer_name == 'SGD_mom':
        optimizer = optim.SGD([x], lr=0.1, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training loop
    losses = []
    start_time = time.time()
    
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = quadratic_loss(x, A, b)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (i + 1) % (num_iters // 5) == 0 or i == 0:
            print(f"[Synthetic-Quadratic] {optimizer_name} - Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    # Rosenbrock function optimization (if enabled)
    if config.get('run_rosenbrock', True):
        # Reset parameters
        x = torch.tensor([0.0, 0.0], requires_grad=True)
        
        # Define the Rosenbrock loss function
        def rosenbrock_loss(x):
            a = 1.0
            b = 100.0
            return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
        
        # Reinitialize optimizer
        if optimizer_name == 'ACM':
            optimizer = ACMOptimizer([x], lr=0.001, beta=0.9, curvature_influence=0.05)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam([x], lr=0.001)
        elif optimizer_name == 'SGD_mom':
            optimizer = optim.SGD([x], lr=0.001, momentum=0.9)
        
        # Training loop for Rosenbrock
        rosenbrock_losses = []
        rosenbrock_start_time = time.time()
        
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = rosenbrock_loss(x)
            loss.backward()
            optimizer.step()
            rosenbrock_losses.append(loss.item())
            
            if (i + 1) % (num_iters // 5) == 0 or i == 0:
                print(f"[Synthetic-Rosenbrock] {optimizer_name} - Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
        
        rosenbrock_training_time = time.time() - rosenbrock_start_time
    else:
        rosenbrock_losses = []
        rosenbrock_training_time = 0
    
    return {
        'optimizer': optimizer_name,
        'quadratic': {
            'losses': losses,
            'final_loss': losses[-1],
            'training_time': training_time,
            'final_x': x.detach().numpy() if not config.get('run_rosenbrock', True) else None
        },
        'rosenbrock': {
            'losses': rosenbrock_losses,
            'final_loss': rosenbrock_losses[-1] if rosenbrock_losses else None,
            'training_time': rosenbrock_training_time,
            'final_x': x.detach().numpy() if config.get('run_rosenbrock', True) else None
        }
    }

def train_cifar10(optimizer_name, data_loaders, config):
    """
    Train a CNN model on CIFAR-10 dataset.
    
    Args:
        optimizer_name (str): Name of the optimizer to use
        data_loaders (tuple): Tuple containing (train_loader, test_loader)
        config (dict): Configuration parameters
        
    Returns:
        tuple: (model, training_history) containing the trained model and training metrics
    """
    # Extract parameters from config
    num_epochs = config.get('cifar10_epochs', 10)
    learning_rate = config.get('cifar10_lr', 0.01)
    quick_test = config.get('quick_test', False)
    
    if quick_test:
        num_epochs = min(num_epochs, 1)  # Limit epochs for quick testing
    
    # Unpack data loaders
    train_loader, test_loader = data_loaders
    
    # Initialize model
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer based on name
    if optimizer_name == 'ACM':
        optimizer = ACMOptimizer(
            model.parameters(), 
            lr=learning_rate, 
            beta=0.9, 
            curvature_influence=0.05
        )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD_mom':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print progress
            if (i + 1) % 100 == 0 and not quick_test:
                print(f'[CIFAR10] {optimizer_name} - Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, '
                      f'Loss: {running_loss/(i+1):.4f}, Acc: {100.*correct/total:.2f}%')
            
            # For quick test, only process a few batches
            if quick_test and i >= 5:
                break
        
        # Calculate epoch statistics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, quick_test)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f'[CIFAR10] {optimizer_name} - Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')
    
    return model, history

def train_mnist(optimizer_name, data_loaders, config):
    """
    Train a model on MNIST dataset for ablation study.
    
    Args:
        optimizer_name (str): Name of the optimizer to use
        data_loaders (tuple): Tuple containing (train_loader, test_loader)
        config (dict): Configuration parameters
        
    Returns:
        tuple: (model, training_history) containing the trained model and training metrics
    """
    # Extract parameters from config
    num_epochs = config.get('mnist_epochs', 5)
    learning_rate = config.get('mnist_lr', 0.01)
    quick_test = config.get('quick_test', False)
    
    if quick_test:
        num_epochs = min(num_epochs, 1)  # Limit epochs for quick testing
    
    # Unpack data loaders
    train_loader, test_loader = data_loaders
    
    # Initialize model
    model = MNISTNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer based on name and hyperparameters
    if optimizer_name == 'ACM':
        # Extract ACM-specific hyperparameters
        beta = config.get('acm_beta', 0.9)
        curvature_influence = config.get('acm_curvature_influence', 0.1)
        
        optimizer = ACMOptimizer(
            model.parameters(), 
            lr=learning_rate, 
            beta=beta, 
            curvature_influence=curvature_influence
        )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD_mom':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print progress
            if (i + 1) % 100 == 0 and not quick_test:
                print(f'[MNIST] {optimizer_name} - Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, '
                      f'Loss: {running_loss/(i+1):.4f}, Acc: {100.*correct/total:.2f}%')
            
            # For quick test, only process a few batches
            if quick_test and i >= 5:
                break
        
        # Calculate epoch statistics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, quick_test)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f'[MNIST] {optimizer_name} - Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')
    
    return model, history

def evaluate_model(model, test_loader, criterion, device, quick_test=False):
    """
    Evaluate a model on the test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for computation
        quick_test (bool): Whether to run a quick test with limited batches
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # For quick test, only process a few batches
            if quick_test and i >= 5:
                break
    
    # Calculate average loss and accuracy
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def save_model(model, optimizer_name, dataset_name, config):
    """
    Save a trained model.
    
    Args:
        model (nn.Module): Model to save
        optimizer_name (str): Name of the optimizer used
        dataset_name (str): Name of the dataset
        config (dict): Configuration parameters
    """
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f'models/{dataset_name}_{optimizer_name}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def train_models(data, config):
    """
    Main training function that trains models for all experiments.
    
    Args:
        data (dict): Dictionary containing preprocessed data
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing all training results
    """
    results = {}
    optimizers = ['ACM', 'Adam', 'SGD_mom']
    
    # Run synthetic experiments
    if 'synthetic' in data and config.get('run_synthetic', True):
        synthetic_results = []
        
        for optimizer_name in optimizers:
            print(f"\n=== Running Synthetic Experiment with {optimizer_name} ===")
            result = train_synthetic_function(optimizer_name, data['synthetic'], config)
            synthetic_results.append(result)
        
        results['synthetic'] = synthetic_results
    
    # Run CIFAR-10 experiments
    if 'cifar10' in data and config.get('run_cifar10', True):
        cifar10_results = {}
        
        for optimizer_name in optimizers:
            print(f"\n=== Running CIFAR-10 Experiment with {optimizer_name} ===")
            model, history = train_cifar10(optimizer_name, data['cifar10'], config)
            
            # Save model if specified in config
            if config.get('save_models', True):
                save_model(model, optimizer_name, 'cifar10', config)
            
            cifar10_results[optimizer_name] = {
                'history': history,
                'final_train_acc': history['train_acc'][-1],
                'final_test_acc': history['test_acc'][-1],
                'final_train_loss': history['train_loss'][-1],
                'final_test_loss': history['test_loss'][-1],
                'total_time': sum(history['epoch_times'])
            }
        
        results['cifar10'] = cifar10_results
    
    # Run MNIST experiments (ablation study)
    if 'mnist' in data and config.get('run_mnist', True):
        mnist_results = {}
        
        # Define different hyperparameter settings for ACM
        acm_configs = []
        
        if config.get('run_ablation', True):
            # Only run ablation if specified in config
            base_config = config.copy()
            
            # Vary beta
            for beta in [0.8, 0.9, 0.95]:
                # Vary curvature influence
                for ci in [0.01, 0.05, 0.1, 0.2]:
                    ablation_config = base_config.copy()
                    ablation_config['acm_beta'] = beta
                    ablation_config['acm_curvature_influence'] = ci
                    acm_configs.append((f'ACM_b{beta}_ci{ci}', ablation_config))
        else:
            # Just use default ACM config
            acm_configs = [('ACM', config)]
        
        # Train with different ACM configurations
        for acm_name, acm_config in acm_configs:
            print(f"\n=== Running MNIST Experiment with {acm_name} ===")
            model, history = train_mnist(acm_name, data['mnist'], acm_config)
            
            # Save model if specified in config
            if config.get('save_models', True):
                save_model(model, acm_name, 'mnist', config)
            
            mnist_results[acm_name] = {
                'history': history,
                'final_train_acc': history['train_acc'][-1],
                'final_test_acc': history['test_acc'][-1],
                'final_train_loss': history['train_loss'][-1],
                'final_test_loss': history['test_loss'][-1],
                'total_time': sum(history['epoch_times']),
                'config': {
                    'beta': acm_config.get('acm_beta', 0.9),
                    'curvature_influence': acm_config.get('acm_curvature_influence', 0.1)
                }
            }
        
        # Train with standard optimizers for comparison
        for optimizer_name in ['Adam', 'SGD_mom']:
            print(f"\n=== Running MNIST Experiment with {optimizer_name} ===")
            model, history = train_mnist(optimizer_name, data['mnist'], config)
            
            # Save model if specified in config
            if config.get('save_models', True):
                save_model(model, optimizer_name, 'mnist', config)
            
            mnist_results[optimizer_name] = {
                'history': history,
                'final_train_acc': history['train_acc'][-1],
                'final_test_acc': history['test_acc'][-1],
                'final_train_loss': history['train_loss'][-1],
                'final_test_loss': history['test_loss'][-1],
                'total_time': sum(history['epoch_times'])
            }
        
        results['mnist'] = mnist_results
    
    return results

if __name__ == "__main__":
    # Simple test to verify the training works
    import torch.nn as nn
    
    # Create a simple model and data
    model = nn.Linear(10, 1)
    data = torch.randn(100, 10)
    target = torch.randn(100, 1)
    
    # Create optimizer
    optimizer = ACMOptimizer(model.parameters(), lr=0.01)
    
    # Test a few optimization steps
    for i in range(5):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        print(f"Step {i+1}, Loss: {loss.item():.4f}")
    
    print("Training test completed successfully.")
