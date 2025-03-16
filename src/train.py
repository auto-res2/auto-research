"""
Model training module for ACM optimizer experiments.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import time
import sys
from tqdm import tqdm
import torchvision.models as models
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

# Add the project root directory to the Python path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Try relative imports (when imported as a module)
    from .utils.optimizers import ACM, ACMNumpy
    from .utils.baseline_optimizers import SGDOptimizer_NP, AdamOptimizer_NP, AdaBeliefOptimizer_NP
    from .utils.synthetic_functions import quadratic_func, rosenbrock_func, rastrigin_func
    from .utils.experiment_utils import set_seed, get_device, save_model, plot_learning_curves, plot_optimizer_comparison, ExperimentLogger
    from .preprocess import preprocess_data
except ImportError:
    # Fall back to absolute imports (when run as a script)
    from utils.optimizers import ACM, ACMNumpy
    from utils.baseline_optimizers import SGDOptimizer_NP, AdamOptimizer_NP, AdaBeliefOptimizer_NP
    from utils.synthetic_functions import quadratic_func, rosenbrock_func, rastrigin_func
    from utils.experiment_utils import set_seed, get_device, save_model, plot_learning_curves, plot_optimizer_comparison, ExperimentLogger
    from preprocess import preprocess_data


def train_synthetic_functions(data, config, logger):
    """
    Train optimizers on synthetic functions.
    
    Args:
        data (dict): Dictionary containing synthetic function data
        config (dict): Configuration dictionary
        logger (ExperimentLogger): Logger for experiment results
        
    Returns:
        dict: Results dictionary
    """
    logger.log("Starting synthetic function optimization...")
    
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Initialize optimizers
    optimizers = {
        "ACM": ACMNumpy(**config["optimizers"]["ACM"]),
        "Adam": AdamOptimizer_NP(**config["optimizers"]["Adam"]),
        "AdaBelief": AdaBeliefOptimizer_NP(**config["optimizers"]["AdaBelief"])
    }
    
    # Function names and their corresponding functions
    functions = {
        "quadratic": quadratic_func,
        "rosenbrock": rosenbrock_func,
        "rastrigin": rastrigin_func
    }
    
    # Results dictionary
    all_results = {}
    
    # Run optimization for each function
    for func_name, func in functions.items():
        logger.log(f"\nOptimizing {func_name} function...")
        
        # Get function-specific data and parameters
        func_data = data[func_name]
        init_x = func_data["init_x"]
        n_iterations = config["functions"][func_name]["n_iterations"]
        
        # Results for this function
        results = {opt_name: {"f_vals": [], "lr_trace": []} for opt_name in optimizers}
        
        # Run optimization with each optimizer
        for opt_name, optimizer in optimizers.items():
            logger.log(f"Optimizer: {opt_name}")
            
            # Initialize x
            x = init_x.copy()
            
            # Optimization loop
            for i in range(n_iterations):
                # Compute function value and gradient
                if func_name == "quadratic":
                    f_val, grad = func(x, func_data["A"], func_data["b"])
                elif func_name == "rosenbrock":
                    f_val, grad = func(x, func_data["a"], func_data["b"])
                else:  # rastrigin
                    f_val, grad = func(x, func_data["A"])
                
                # Update parameters
                x, lr = optimizer.update(x, grad)
                
                # Store results
                results[opt_name]["f_vals"].append(f_val)
                results[opt_name]["lr_trace"].append(lr)
                
                # Log progress
                if i % max(1, n_iterations // 10) == 0:
                    logger.log(f"  Iteration {i:4d}: f(x) = {f_val:.6f}, lr = {lr:.6e}")
            
            # Log final result
            logger.log(f"  Final: f(x) = {results[opt_name]['f_vals'][-1]:.6f}")
        
        # Store results for this function
        all_results[func_name] = results
        
        # Plot results
        plot_path = os.path.join(config["output_dir"], f"{func_name}_optimization.png")
        plot_optimizer_comparison(results, func_name, plot_path)
    
    logger.log("Synthetic function optimization completed.")
    return all_results


def train_cifar10_resnet(train_loader, test_loader, config, logger):
    """
    Train ResNet-18 on CIFAR-10 dataset.
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config (dict): Configuration dictionary
        logger (ExperimentLogger): Logger for experiment results
        
    Returns:
        dict: Results dictionary
    """
    logger.log("Starting CIFAR-10 ResNet-18 training...")
    
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Get device
    device = get_device()
    
    # Create model
    model = models.resnet18(num_classes=config["model"]["num_classes"])
    model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Results dictionary
    results = {}
    
    # Train with each optimizer
    for opt_name, opt_config in config["training"]["optimizers"].items():
        logger.log(f"\nTraining with optimizer: {opt_name}")
        
        # Reset model
        model = models.resnet18(num_classes=config["model"]["num_classes"])
        model.to(device)
        
        # Initialize optimizer
        if opt_name == "ACM":
            optimizer = ACM(model.parameters(), **opt_config)
        elif opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), **opt_config)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        # Initialize scheduler
        scheduler_config = config["training"]["scheduler"]
        if scheduler_config["type"] == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"]
            )
        else:
            scheduler = None
        
        # Training metrics
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Training loop
        for epoch in range(config["training"]["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # If test_mode, only run a few batches
                if config["test_mode"] and batch_idx >= 2:
                    break
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Calculate training metrics
            avg_train_loss = train_loss / (batch_idx + 1)
            train_accuracy = 100.0 * correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # If test_mode, only run a few batches
                    if config["test_mode"] and batch_idx >= 2:
                        break
            
            # Calculate validation metrics
            avg_val_loss = val_loss / (batch_idx + 1)
            val_accuracy = 100.0 * correct / total
            
            # Store metrics
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Log metrics
            logger.log_metrics(epoch + 1, {
                "train_loss": avg_train_loss,
                "train_acc": train_accuracy,
                "val_loss": avg_val_loss,
                "val_acc": val_accuracy
            })
        
        # Save model
        if config["training"]["save_model"]:
            save_path = os.path.join(config["training"]["save_dir"], f"resnet18_{opt_name}.pth")
            save_model(model, optimizer, config["training"]["epochs"], val_losses[-1], val_accuracies[-1], save_path)
        
        # Plot learning curves
        plot_path = os.path.join(config["output_dir"], f"cifar10_resnet18_{opt_name}.png")
        plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, f"CIFAR-10 ResNet-18 ({opt_name})", plot_path)
        
        # Store results
        results[opt_name] = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "final_val_loss": val_losses[-1],
            "final_val_accuracy": val_accuracies[-1]
        }
    
    logger.log("CIFAR-10 ResNet-18 training completed.")
    return results


def train_transformer_lm(dataloader, tokenizer, config, logger):
    """
    Train transformer language model.
    
    Args:
        dataloader: DataLoader for training data
        tokenizer: Tokenizer for the model
        config (dict): Configuration dictionary
        logger (ExperimentLogger): Logger for experiment results
        
    Returns:
        dict: Results dictionary
    """
    logger.log("Starting transformer language model training...")
    
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Get device
    device = get_device()
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(config["model"]["name"])
    model.to(device)
    
    # Initialize optimizer
    optimizer_config = config["training"]["optimizer"]
    if optimizer_config["name"] == "ACM":
        optimizer = ACM(
            model.parameters(),
            lr=optimizer_config["lr"],
            beta=optimizer_config["beta"],
            weight_decay=optimizer_config["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    
    # Initialize scheduler
    scheduler_config = config["training"]["scheduler"]
    if scheduler_config["type"] == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_config["warmup_steps"],
            num_training_steps=scheduler_config["total_steps"]
        )
    else:
        scheduler = None
    
    # Training metrics
    losses = []
    perplexities = []
    
    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += loss.item()
            
            # If test_mode, only run a few batches
            if config["test_mode"] and batch_idx >= 2:
                break
        
        # Calculate metrics
        avg_loss = epoch_loss / (batch_idx + 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Store metrics
        losses.append(avg_loss)
        perplexities.append(perplexity)
        
        # Log metrics
        logger.log_metrics(epoch + 1, {
            "loss": avg_loss,
            "perplexity": perplexity
        })
    
    # Save model
    if config["training"]["save_model"]:
        save_path = os.path.join(config["training"]["save_dir"], "transformer_lm.pth")
        save_model(model, optimizer, config["training"]["epochs"], losses[-1], 0.0, save_path)
    
    # Store results
    results = {
        "losses": losses,
        "perplexities": perplexities,
        "final_loss": losses[-1],
        "final_perplexity": perplexities[-1]
    }
    
    logger.log("Transformer language model training completed.")
    return results


def train_model(config_path, experiment_type):
    """
    Train model based on experiment type.
    
    Args:
        config_path (str): Path to configuration file
        experiment_type (str): Type of experiment ('synthetic', 'cifar10', or 'transformer')
        
    Returns:
        dict: Results dictionary
    """
    # Preprocess data
    data, config = preprocess_data(config_path, experiment_type)
    
    # Create logger
    logger = ExperimentLogger(config["output_dir"], config["experiment_name"])
    logger.log("Starting model training...")
    
    # Train model based on experiment type
    if experiment_type == 'synthetic':
        results = train_synthetic_functions(data, config, logger)
    
    elif experiment_type == 'cifar10':
        train_loader, test_loader = data
        results = train_cifar10_resnet(train_loader, test_loader, config, logger)
    
    elif experiment_type == 'transformer':
        dataloader, tokenizer = data
        results = train_transformer_lm(dataloader, tokenizer, config, logger)
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    logger.log("Model training completed.")
    return results, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training for ACM optimizer experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True, choices=['synthetic', 'cifar10', 'transformer'], help='Type of experiment')
    
    args = parser.parse_args()
    
    # Train model
    results, config = train_model(args.config, args.experiment)
    
    print(f"Model training for {args.experiment} experiment completed.")
