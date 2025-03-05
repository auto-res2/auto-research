import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# Use relative imports when running from within the package
try:
    # When running as a module (e.g., python -m src.train)
    from src.utils.optimizers import ACMOptimizer
except ModuleNotFoundError:
    # When running directly (e.g., python src/train.py)
    from utils.optimizers import ACMOptimizer

class SimpleCNN(nn.Module):
    """Simple CNN model for CIFAR-10 classification."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class MNISTNet(nn.Module):
    """Simple network for MNIST classification."""
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
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def quadratic_loss(x, A, b):
    """Compute the quadratic loss: f(x) = 0.5 * x^T A x - b^T x."""
    return 0.5 * torch.matmul(torch.matmul(x, A), x) - torch.matmul(b, x)

def rosenbrock_loss(x, a=1.0, b=100.0):
    """Compute the Rosenbrock loss: f(x) = (a - x[0])^2 + b * (x[1] - x[0]^2)^2."""
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def train_synthetic(data, optimizer_type, config):
    """
    Train on synthetic optimization problems.
    
    Args:
        data: Dictionary containing synthetic datasets
        optimizer_type: Type of optimizer to use ('ACM', 'Adam', or 'SGD_mom')
        config: Configuration dictionary with training parameters
        
    Returns:
        Dictionary containing training results
    """
    results = {
        'quadratic': {'losses': []},
        'rosenbrock': {'losses': []}
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Train on quadratic function
    print(f"Training on quadratic function with {optimizer_type}...")
    x = torch.tensor(data['quadratic']['data'][0], requires_grad=True)
    A = data['quadratic']['A']
    b = data['quadratic']['b']
    
    # Initialize optimizer
    if optimizer_type == 'ACM':
        optimizer = ACMOptimizer([x], lr=config['lr'], beta=config['beta'], 
                                curvature_influence=config['curvature_influence'])
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam([x], lr=config['lr'])
    else:  # SGD with momentum
        optimizer = optim.SGD([x], lr=config['lr'], momentum=config['beta'])
    
    # Training loop
    for i in range(config['num_iters']):
        optimizer.zero_grad()
        loss = quadratic_loss(x, A, b)
        loss.backward()
        optimizer.step()
        results['quadratic']['losses'].append(loss.item())
        
        if (i + 1) % (config['num_iters'] // 5) == 0 or i == 0:
            print(f"Iter {i+1}/{config['num_iters']} - Loss: {loss.item():.4f}")
    
    # Train on Rosenbrock function
    print(f"\nTraining on Rosenbrock function with {optimizer_type}...")
    x = torch.tensor(data['rosenbrock']['data'][0], requires_grad=True)
    a = data['rosenbrock']['a']
    b = data['rosenbrock']['b']
    
    # Initialize optimizer
    if optimizer_type == 'ACM':
        optimizer = ACMOptimizer([x], lr=config['lr'], beta=config['beta'], 
                                curvature_influence=config['curvature_influence'])
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam([x], lr=config['lr'])
    else:  # SGD with momentum
        optimizer = optim.SGD([x], lr=config['lr'], momentum=config['beta'])
    
    # Training loop
    for i in range(config['num_iters']):
        optimizer.zero_grad()
        loss = rosenbrock_loss(x, a, b)
        loss.backward()
        optimizer.step()
        results['rosenbrock']['losses'].append(loss.item())
        
        if (i + 1) % (config['num_iters'] // 5) == 0 or i == 0:
            print(f"Iter {i+1}/{config['num_iters']} - Loss: {loss.item():.4f}")
    
    return results

def train_cifar10(data_loaders, optimizer_type, config):
    """
    Train a CNN model on CIFAR-10 dataset.
    
    Args:
        data_loaders: Dictionary containing train and test data loaders
        optimizer_type: Type of optimizer to use ('ACM', 'Adam', or 'SGD_mom')
        config: Configuration dictionary with training parameters
        
    Returns:
        Dictionary containing trained model and training results
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Initialize model
    model = SimpleCNN().to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    if optimizer_type == 'ACM':
        optimizer = ACMOptimizer(model.parameters(), lr=config['lr'], beta=config['beta'], 
                                curvature_influence=config['curvature_influence'],
                                weight_decay=config['weight_decay'])
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:  # SGD with momentum
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['beta'], 
                             weight_decay=config['weight_decay'])
    
    # Initialize results dictionary
    results = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': []
    }
    
    # Training loop
    print(f"Training CIFAR-10 model with {optimizer_type}...")
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(data_loaders['train_loader']):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0 and config['verbose']:
                print(f"Epoch {epoch+1}/{config['num_epochs']}, Batch {i+1}/{len(data_loaders['train_loader'])}, "
                      f"Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0
        
        # Calculate training metrics
        train_loss = running_loss / len(data_loaders['train_loader'])
        train_acc = 100. * correct / total
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loaders['test_loader']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate test metrics
        test_loss = test_loss / len(data_loaders['test_loader'])
        test_acc = 100. * correct / total
        results['test_losses'].append(test_loss)
        results['test_accs'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f"models/cifar10_{optimizer_type.lower()}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return {
        'model': model,
        'results': results
    }

def train_mnist(data_loaders, optimizer_type, config):
    """
    Train a model on MNIST dataset.
    
    Args:
        data_loaders: Dictionary containing train and test data loaders
        optimizer_type: Type of optimizer to use ('ACM', 'Adam', or 'SGD_mom')
        config: Configuration dictionary with training parameters
        
    Returns:
        Dictionary containing trained model and training results
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Initialize model
    model = MNISTNet().to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    if optimizer_type == 'ACM':
        optimizer = ACMOptimizer(model.parameters(), lr=config['lr'], beta=config['beta'], 
                                curvature_influence=config['curvature_influence'],
                                weight_decay=config['weight_decay'])
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:  # SGD with momentum
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['beta'], 
                             weight_decay=config['weight_decay'])
    
    # Initialize results dictionary
    results = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': []
    }
    
    # Training loop
    print(f"Training MNIST model with {optimizer_type}...")
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(data_loaders['train_loader']):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0 and config['verbose']:
                print(f"Epoch {epoch+1}/{config['num_epochs']}, Batch {i+1}/{len(data_loaders['train_loader'])}, "
                      f"Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0
        
        # Calculate training metrics
        train_loss = running_loss / len(data_loaders['train_loader'])
        train_acc = 100. * correct / total
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loaders['test_loader']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate test metrics
        test_loss = test_loss / len(data_loaders['test_loader'])
        test_acc = 100. * correct / total
        results['test_losses'].append(test_loss)
        results['test_accs'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f"models/mnist_{optimizer_type.lower()}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return {
        'model': model,
        'results': results
    }

def run_ablation_study(data_loaders, config):
    """
    Run ablation study on MNIST dataset with different hyperparameters for ACM optimizer.
    
    Args:
        data_loaders: Dictionary containing train and test data loaders
        config: Configuration dictionary with training parameters
        
    Returns:
        Dictionary containing ablation study results
    """
    print("Running ablation study on MNIST dataset...")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Define hyperparameter combinations to test
    lr_values = config['lr_values']
    beta_values = config['beta_values']
    curvature_influence_values = config['curvature_influence_values']
    
    # Initialize results dictionary
    results = {}
    
    # Run experiments with different learning rates
    print("Testing different learning rates...")
    lr_results = []
    for lr in lr_values:
        model = MNISTNet().to(device)
        optimizer = ACMOptimizer(model.parameters(), lr=lr, beta=config['beta'], 
                                curvature_influence=config['curvature_influence'])
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        model.train()
        train_losses = []
        
        for epoch in range(config['ablation_epochs']):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loaders['train_loader']):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i >= config['ablation_batches']:
                    break
            
            avg_loss = running_loss / (i + 1)
            train_losses.append(avg_loss)
            print(f"LR={lr}, Epoch {epoch+1}/{config['ablation_epochs']}, Loss: {avg_loss:.4f}")
        
        lr_results.append({
            'lr': lr,
            'losses': train_losses
        })
    
    results['lr_study'] = lr_results
    
    # Run experiments with different beta values
    print("\nTesting different beta values...")
    beta_results = []
    for beta in beta_values:
        model = MNISTNet().to(device)
        optimizer = ACMOptimizer(model.parameters(), lr=config['lr'], beta=beta, 
                                curvature_influence=config['curvature_influence'])
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        model.train()
        train_losses = []
        
        for epoch in range(config['ablation_epochs']):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loaders['train_loader']):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i >= config['ablation_batches']:
                    break
            
            avg_loss = running_loss / (i + 1)
            train_losses.append(avg_loss)
            print(f"Beta={beta}, Epoch {epoch+1}/{config['ablation_epochs']}, Loss: {avg_loss:.4f}")
        
        beta_results.append({
            'beta': beta,
            'losses': train_losses
        })
    
    results['beta_study'] = beta_results
    
    # Run experiments with different curvature influence values
    print("\nTesting different curvature influence values...")
    curvature_results = []
    for ci in curvature_influence_values:
        model = MNISTNet().to(device)
        optimizer = ACMOptimizer(model.parameters(), lr=config['lr'], beta=config['beta'], 
                                curvature_influence=ci)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        model.train()
        train_losses = []
        
        for epoch in range(config['ablation_epochs']):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loaders['train_loader']):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i >= config['ablation_batches']:
                    break
            
            avg_loss = running_loss / (i + 1)
            train_losses.append(avg_loss)
            print(f"CI={ci}, Epoch {epoch+1}/{config['ablation_epochs']}, Loss: {avg_loss:.4f}")
        
        curvature_results.append({
            'curvature_influence': ci,
            'losses': train_losses
        })
    
    results['curvature_study'] = curvature_results
    
    return results

def quick_test():
    """Run a quick test with minimal iterations to verify code execution."""
    from src.preprocess import preprocess_data
    
    print("Running quick test...")
    
    # Set minimal configuration
    config = {
        'synthetic': {
            'n_samples': 10,
            'seed': 42
        },
        'cifar10': {
            'batch_size': 64,
            'download': True
        },
        'mnist': {
            'batch_size': 64,
            'download': True
        },
        'train': {
            'seed': 42,
            'num_iters': 5,
            'num_epochs': 1,
            'lr': 0.01,
            'beta': 0.9,
            'curvature_influence': 0.1,
            'weight_decay': 0.0001,
            'verbose': False
        },
        'ablation': {
            'seed': 42,
            'ablation_epochs': 1,
            'ablation_batches': 5,
            'lr': 0.01,
            'beta': 0.9,
            'curvature_influence': 0.1,
            'lr_values': [0.001, 0.01],
            'beta_values': [0.8, 0.9],
            'curvature_influence_values': [0.05, 0.1]
        }
    }
    
    # Preprocess data
    data = preprocess_data(config)
    
    # Test synthetic optimization
    print("\nTesting synthetic optimization...")
    train_synthetic(data['synthetic'], 'ACM', config['train'])
    
    # Test CIFAR-10 training (limited to 5 batches)
    print("\nTesting CIFAR-10 training...")
    cifar_train_loader = data['cifar10']['train_loader']
    cifar_test_loader = data['cifar10']['test_loader']
    
    # Create a smaller dataset for quick testing
    class LimitedLoader:
        def __init__(self, loader, limit):
            self.loader = loader
            self.limit = limit
            
        def __iter__(self):
            counter = 0
            for item in self.loader:
                if counter >= self.limit:
                    break
                counter += 1
                yield item
                
        def __len__(self):
            return min(self.limit, len(self.loader))
    
    limited_cifar_train = LimitedLoader(cifar_train_loader, 5)
    limited_cifar_test = LimitedLoader(cifar_test_loader, 5)
    
    train_cifar10({
        'train_loader': limited_cifar_train,
        'test_loader': limited_cifar_test
    }, 'ACM', config['train'])
    
    # Test MNIST training (limited to 5 batches)
    print("\nTesting MNIST training...")
    mnist_train_loader = data['mnist']['train_loader']
    mnist_test_loader = data['mnist']['test_loader']
    
    limited_mnist_train = LimitedLoader(mnist_train_loader, 5)
    limited_mnist_test = LimitedLoader(mnist_test_loader, 5)
    
    train_mnist({
        'train_loader': limited_mnist_train,
        'test_loader': limited_mnist_test
    }, 'ACM', config['train'])
    
    # Test ablation study (very limited)
    print("\nTesting ablation study...")
    run_ablation_study({
        'train_loader': limited_mnist_train,
        'test_loader': limited_mnist_test
    }, config['ablation'])
    
    print("\nQuick test completed successfully!")

if __name__ == "__main__":
    # Run quick test to verify code execution
    quick_test()
