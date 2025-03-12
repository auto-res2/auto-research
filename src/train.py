"""Training code for experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.optimizers import ACMOptimizer
from src.utils.models import SimpleCNN, MNISTNet
from src.utils.utils import set_seed, get_device, plot_loss_curves, log_experiment_results
from config.experiment_config import (
    RANDOM_SEED,
    DEVICE,
    SYNTHETIC_ITERS,
    SYNTHETIC_QUICK_ITERS,
    CIFAR_EPOCHS,
    CIFAR_QUICK_EPOCHS,
    MNIST_EPOCHS,
    MNIST_QUICK_EPOCHS,
    OPTIMIZERS,
    QUICK_TEST,
)


def train_synthetic(quick_test=QUICK_TEST):
    """Run synthetic optimization benchmark.
    
    Args:
        quick_test (bool): Whether to run a quick test with minimal iterations
        
    Returns:
        dict: Results of the synthetic optimization benchmark
    """
    print("\n=== Synthetic Optimization Benchmark ===")
    set_seed(RANDOM_SEED)
    device = get_device()
    
    # Number of iterations
    num_iters = SYNTHETIC_QUICK_ITERS if quick_test else SYNTHETIC_ITERS
    
    # Example 2D quadratic. A is positive definite.
    A = torch.tensor([[3.0, 0.2], [0.2, 2.0]], device=device)
    b = torch.tensor([1.0, 1.0], device=device)
    
    # Define the quadratic loss function
    def quadratic_loss(x):
        return 0.5 * x @ A @ x - b @ x
    
    # Prepare optimizers
    optimizers_dict = {
        "ACM": lambda params: ACMOptimizer(
            params, 
            lr=OPTIMIZERS["ACM"]["lr"],
            beta=OPTIMIZERS["ACM"]["beta"],
            curvature_influence=OPTIMIZERS["ACM"]["curvature_influence"]
        ),
        "Adam": lambda params: optim.Adam(
            params, 
            lr=OPTIMIZERS["Adam"]["lr"],
            betas=OPTIMIZERS["Adam"]["betas"]
        ),
        "SGD_momentum": lambda params: optim.SGD(
            params, 
            lr=OPTIMIZERS["SGD_momentum"]["lr"],
            momentum=OPTIMIZERS["SGD_momentum"]["momentum"]
        )
    }
    
    results = {name: [] for name in optimizers_dict.keys()}
    final_positions = {}
    
    # Run separate optimization runs for each optimizer
    for name, opt_class in optimizers_dict.items():
        print(f"\nRunning optimization with {name}")
        
        # Reinitialize the initial point for fairness
        x_data = torch.randn(2, requires_grad=True, device=device)
        
        # Instantiate the optimizer
        optimizer = opt_class([x_data])
        
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = quadratic_loss(x_data)
            loss.backward()
            optimizer.step()
            results[name].append(loss.item())
            
            if (i + 1) % (num_iters // 5) == 0 or i == 0:
                print(f"Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
        
        final_positions[name] = x_data.detach().cpu().numpy()
    
    # Plot the convergence curves
    plot_loss_curves(
        results, 
        "Quadratic Function Optimization",
        save_path="./logs/synthetic_quadratic.png"
    )
    
    # Run Rosenbrock function optimization
    print("\n=== Rosenbrock Function Optimization ===")
    
    # Define the Rosenbrock loss function
    def rosenbrock_loss(x):
        a = 1.0
        b = 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    rosenbrock_results = {name: [] for name in optimizers_dict.keys()}
    rosenbrock_final_positions = {}
    
    for name, opt_class in optimizers_dict.items():
        print(f"\nRunning optimization with {name}")
        
        # Reinitialize the initial point for fairness
        x_data = torch.tensor([0.0, 0.0], requires_grad=True, device=device)
        
        # Instantiate the optimizer
        optimizer = opt_class([x_data])
        
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = rosenbrock_loss(x_data)
            loss.backward()
            optimizer.step()
            rosenbrock_results[name].append(loss.item())
            
            if (i + 1) % (num_iters // 5) == 0 or i == 0:
                print(f"Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
        
        rosenbrock_final_positions[name] = x_data.detach().cpu().numpy()
    
    # Plot the convergence curves
    plot_loss_curves(
        rosenbrock_results, 
        "Rosenbrock Function Optimization",
        save_path="./logs/synthetic_rosenbrock.png"
    )
    
    return {
        "quadratic": {
            "losses": results,
            "final_positions": final_positions
        },
        "rosenbrock": {
            "losses": rosenbrock_results,
            "final_positions": rosenbrock_final_positions
        }
    }


def train_cifar10(train_loader, test_loader, quick_test=QUICK_TEST):
    """Train a CNN on CIFAR-10.
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        quick_test (bool): Whether to run a quick test with minimal epochs
        
    Returns:
        dict: Results of the CIFAR-10 training
    """
    print("\n=== CIFAR-10 CNN Training ===")
    set_seed(RANDOM_SEED)
    device = get_device()
    
    # Number of epochs
    num_epochs = CIFAR_QUICK_EPOCHS if quick_test else CIFAR_EPOCHS
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Prepare optimizers
    optimizers_dict = {
        "ACM": lambda params: ACMOptimizer(
            params, 
            lr=OPTIMIZERS["ACM"]["lr"],
            beta=OPTIMIZERS["ACM"]["beta"],
            curvature_influence=OPTIMIZERS["ACM"]["curvature_influence"]
        ),
        "Adam": lambda params: optim.Adam(
            params, 
            lr=OPTIMIZERS["Adam"]["lr"],
            betas=OPTIMIZERS["Adam"]["betas"]
        ),
        "SGD_momentum": lambda params: optim.SGD(
            params, 
            lr=OPTIMIZERS["SGD_momentum"]["lr"],
            momentum=OPTIMIZERS["SGD_momentum"]["momentum"]
        )
    }
    
    results = {name: {"train_losses": [], "test_accuracies": []} for name in optimizers_dict.keys()}
    
    # Train with each optimizer
    for name, opt_class in optimizers_dict.items():
        print(f"\nTraining with {name} optimizer")
        
        # Initialize model
        model = SimpleCNN().to(device)
        
        # Initialize optimizer
        optimizer = opt_class(model.parameters())
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            # Training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": train_loss / (progress_bar.n + 1)})
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            results[name]["train_losses"].append(avg_train_loss)
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_accuracy = 100.0 * correct / total
            results[name]["test_accuracies"].append(test_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Save model
        torch.save(model.state_dict(), f"./models/cifar10_{name}.pth")
    
    # Plot training loss curves
    train_losses = {name: data["train_losses"] for name, data in results.items()}
    plot_loss_curves(
        train_losses, 
        "CIFAR-10 Training Loss",
        save_path="./logs/cifar10_train_loss.png"
    )
    
    # Plot test accuracy curves
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["test_accuracies"], label=name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("CIFAR-10 Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("./logs/cifar10_test_accuracy.png")
    plt.close()
    
    return results


def train_mnist_ablation(train_loader, test_loader, quick_test=QUICK_TEST):
    """Run ablation study on MNIST.
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        quick_test (bool): Whether to run a quick test with minimal epochs
        
    Returns:
        dict: Results of the MNIST ablation study
    """
    print("\n=== MNIST Ablation Study ===")
    set_seed(RANDOM_SEED)
    device = get_device()
    
    # Number of epochs
    num_epochs = MNIST_QUICK_EPOCHS if quick_test else MNIST_EPOCHS
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define different curvature influence values for ablation
    curvature_values = [0.0, 0.01, 0.1, 0.5, 1.0] if not quick_test else [0.0, 0.1, 1.0]
    
    results = {f"ACM_curv_{curv}": {"train_losses": [], "test_accuracies": []} for curv in curvature_values}
    
    # Add Adam and SGD for comparison
    results["Adam"] = {"train_losses": [], "test_accuracies": []}
    results["SGD_momentum"] = {"train_losses": [], "test_accuracies": []}
    
    # Run ablation study
    for name in results.keys():
        print(f"\nTraining with {name}")
        
        # Initialize model
        model = MNISTNet().to(device)
        
        # Initialize optimizer based on name
        if name.startswith("ACM"):
            curv_value = float(name.split("_")[-1])
            optimizer = ACMOptimizer(
                model.parameters(),
                lr=OPTIMIZERS["ACM"]["lr"],
                beta=OPTIMIZERS["ACM"]["beta"],
                curvature_influence=curv_value
            )
        elif name == "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=OPTIMIZERS["Adam"]["lr"],
                betas=OPTIMIZERS["Adam"]["betas"]
            )
        elif name == "SGD_momentum":
            optimizer = optim.SGD(
                model.parameters(),
                lr=OPTIMIZERS["SGD_momentum"]["lr"],
                momentum=OPTIMIZERS["SGD_momentum"]["momentum"]
            )
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            # Training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": train_loss / (progress_bar.n + 1)})
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            results[name]["train_losses"].append(avg_train_loss)
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_accuracy = 100.0 * correct / total
            results[name]["test_accuracies"].append(test_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Save model
        torch.save(model.state_dict(), f"./models/mnist_{name}.pth")
    
    # Plot training loss curves
    train_losses = {name: data["train_losses"] for name, data in results.items()}
    plot_loss_curves(
        train_losses, 
        "MNIST Training Loss (Ablation Study)",
        save_path="./logs/mnist_ablation_train_loss.png"
    )
    
    # Plot test accuracy curves
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["test_accuracies"], label=name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("MNIST Test Accuracy (Ablation Study)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("./logs/mnist_ablation_test_accuracy.png")
    plt.close()
    
    return results
