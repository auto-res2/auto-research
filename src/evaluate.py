#!/usr/bin/env python3
"""
Evaluation module for ACM optimizer experiments.

This module contains functions for evaluating trained models and comparing
the performance of different optimizers, including the ACM optimizer.
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import optuna
from torchvision import models
from src.preprocess import get_device, get_dataloaders
from src.train import get_model, get_optimizer, train_and_evaluate, train_synthetic_function
from src.train import plot_training_results, plot_optimization_trajectories, save_model
from src.utils.optimizers import ACM


def experiment1(config, test_run=False):
    """
    Experiment 1: Convergence Speed and Generalization on CIFAR-10.
    
    This experiment trains a simple CNN model on CIFAR-10 using ACM, SGD, and Adam
    optimizers, and compares their training loss and test accuracy.
    
    Args:
        config (dict): Configuration dictionary.
        test_run (bool): Whether this is a test run with reduced dataset size and epochs.
        
    Returns:
        tuple: (losses_acm, losses_sgd, losses_adam, acc_acm, acc_sgd, acc_adam)
    """
    print("=== Experiment 1: Convergence Speed and Generalization on CIFAR-10 ===")
    
    # Get device
    device = get_device(config)
    
    # Get experiment configuration
    exp_config = config['experiment1']
    if test_run:
        num_epochs = config['test_run']['experiment1']['num_epochs']
        batch_size = config['test_run']['experiment1']['batch_size']
    else:
        num_epochs = exp_config['num_epochs']
        batch_size = exp_config['batch_size']
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(config, 'cifar10', test_run)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize models and optimizers
    def get_fresh_model():
        return get_model(exp_config['model'], num_classes=10, device=device)
    
    # Train with ACM optimizer
    print("\nTraining with ACM optimizer...")
    model_acm = get_fresh_model()
    optimizer_acm = get_optimizer('acm', model_acm.parameters(), config, 'experiment1')
    losses_acm, acc_acm = train_and_evaluate(
        model_acm, optimizer_acm, train_loader, test_loader, criterion, 
        num_epochs, device, optimizer_name="ACM"
    )
    # Convert to lists to ensure compatibility
    losses_acm = list(losses_acm)
    acc_acm = list(acc_acm)
    
    # Save model if configured
    if exp_config.get('save_model', False):
        save_path = os.path.join(config['general']['model_dir'], 'experiment1_acm.pth')
        save_model(model_acm, optimizer_acm, num_epochs, losses_acm[-1], acc_acm[-1], save_path)
    
    # Train with SGD optimizer
    print("\nTraining with SGD optimizer...")
    model_sgd = get_fresh_model()
    optimizer_sgd = get_optimizer('sgd', model_sgd.parameters(), config, 'experiment1')
    losses_sgd, acc_sgd = train_and_evaluate(
        model_sgd, optimizer_sgd, train_loader, test_loader, criterion, 
        num_epochs, device, optimizer_name="SGD"
    )
    # Convert to lists to ensure compatibility
    losses_sgd = list(losses_sgd)
    acc_sgd = list(acc_sgd)
    
    # Save model if configured
    if exp_config.get('save_model', False):
        save_path = os.path.join(config['general']['model_dir'], 'experiment1_sgd.pth')
        save_model(model_sgd, optimizer_sgd, num_epochs, losses_sgd[-1], acc_sgd[-1], save_path)
    
    # Train with Adam optimizer
    print("\nTraining with Adam optimizer...")
    model_adam = get_fresh_model()
    optimizer_adam = get_optimizer('adam', model_adam.parameters(), config, 'experiment1')
    losses_adam, acc_adam = train_and_evaluate(
        model_adam, optimizer_adam, train_loader, test_loader, criterion, 
        num_epochs, device, optimizer_name="Adam"
    )
    # Convert to lists to ensure compatibility
    losses_adam = list(losses_adam)
    acc_adam = list(acc_adam)
    
    # Save model if configured
    if exp_config.get('save_model', False):
        save_path = os.path.join(config['general']['model_dir'], 'experiment1_adam.pth')
        save_model(model_adam, optimizer_adam, num_epochs, losses_adam[-1], acc_adam[-1], save_path)
    
    # Plot results
    if exp_config.get('save_plots', False):
        epochs = list(range(1, num_epochs + 1))
        save_path = os.path.join(config['general']['log_dir'], 'experiment1_results.png')
        plot_training_results(
            epochs, losses_acm, losses_sgd, losses_adam, 
            acc_acm, acc_sgd, acc_adam, save_path
        )
    
    print("=== Experiment 1 Completed ===\n")
    
    return losses_acm, losses_sgd, losses_adam, acc_acm, acc_sgd, acc_adam


def experiment2(config, test_run=False):
    """
    Experiment 2: Adaptive Behavior on Ill-Conditioned Synthetic Loss Landscapes.
    
    This experiment creates a synthetic loss landscape with different curvatures in
    different directions, optimizes the loss using ACM, SGD, and Adam optimizers,
    and plots the optimization trajectories.
    
    Args:
        config (dict): Configuration dictionary.
        test_run (bool): Whether this is a test run with reduced steps.
        
    Returns:
        tuple: (traj_acm, traj_sgd, traj_adam)
    """
    print("=== Experiment 2: Adaptive Behavior on Synthetic Ill-Conditioned Loss Landscapes ===")
    
    # Get device
    device = get_device(config)
    
    # Get experiment configuration
    exp_config = config['experiment2']
    if test_run:
        num_steps = config['test_run']['experiment2']['num_steps']
    else:
        num_steps = exp_config['num_steps']
    
    # Synthetic loss parameters
    a = exp_config['synthetic_loss']['a']  # Curvature in x direction
    b = exp_config['synthetic_loss']['b']  # Curvature in y direction
    
    # Define synthetic loss function
    def synthetic_loss(params):
        x, y = params[0], params[1]
        return a * x**2 + b * y**2
    
    print(f"Synthetic loss function: f(x, y) = {a}*x^2 + {b}*y^2")
    print(f"Optimizing with {num_steps} steps...")
    
    # Optimize with ACM
    print("Optimizing with ACM...")
    lr_acm = exp_config['optimizers']['acm']['lr']
    traj_acm = train_synthetic_function('acm', synthetic_loss, num_steps, lr_acm, device)
    
    # Optimize with SGD
    print("Optimizing with SGD...")
    lr_sgd = exp_config['optimizers']['sgd']['lr']
    traj_sgd = train_synthetic_function('sgd', synthetic_loss, num_steps, lr_sgd, device)
    
    # Optimize with Adam
    print("Optimizing with Adam...")
    lr_adam = exp_config['optimizers']['adam']['lr']
    traj_adam = train_synthetic_function('adam', synthetic_loss, num_steps, lr_adam, device)
    
    print("Trajectories computed.")
    
    # Plot results
    if exp_config.get('save_plots', False):
        save_path = os.path.join(config['general']['log_dir'], 'experiment2_results.png')
        plot_optimization_trajectories(traj_acm, traj_sgd, traj_adam, synthetic_loss, save_path)
    
    print("=== Experiment 2 Completed ===\n")
    
    return traj_acm, traj_sgd, traj_adam


def experiment3(config, test_run=False):
    """
    Experiment 3: Sensitivity Analysis of Adaptive Regularization and Curvature Scaling.
    
    This experiment uses Optuna to perform hyperparameter tuning for the ACM optimizer
    on CIFAR-100, and analyzes the sensitivity of the optimizer to different hyperparameter values.
    
    Args:
        config (dict): Configuration dictionary.
        test_run (bool): Whether this is a test run with reduced dataset size, epochs, and trials.
        
    Returns:
        optuna.study.Study: Completed Optuna study.
    """
    print("=== Experiment 3: Sensitivity Analysis via Hyperparameter Tuning on CIFAR-100 ===")
    
    # Get device
    device = get_device(config)
    
    # Get experiment configuration
    exp_config = config['experiment3']
    if test_run:
        num_epochs = config['test_run']['experiment3']['num_epochs']
        n_trials = config['test_run']['experiment3']['n_trials']
        batch_size = config['test_run']['experiment3']['batch_size']
    else:
        num_epochs = exp_config['num_epochs']
        n_trials = exp_config['n_trials']
        batch_size = exp_config['batch_size']
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(config, 'cifar100', test_run)
    
    # Define objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters
        lr_range = exp_config['acm_param_space']['lr']
        beta_range = exp_config['acm_param_space']['beta']
        weight_decay_range = exp_config['acm_param_space']['weight_decay']
        
        lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
        beta = trial.suggest_float("beta", beta_range[0], beta_range[1])
        weight_decay = trial.suggest_float(
            "weight_decay", weight_decay_range[0], weight_decay_range[1], log=True
        )
        
        print(f"Trial hyperparameters: lr={lr:.5f}, beta={beta:.3f}, weight_decay={weight_decay:.5f}")
        
        # Initialize model and optimizer
        model = get_model(exp_config['model'], num_classes=100, device=device)
        optimizer = ACM(
            model.parameters(), lr=lr, beta=beta, weight_decay=weight_decay
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train for specified number of epochs
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f"Trial epoch {epoch+1}/{num_epochs} completed.")
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = correct / total
        print(f"Trial validation accuracy: {val_acc:.4f}")
        
        # Save model if configured
        if exp_config.get('save_model', False):
            save_path = os.path.join(
                config['general']['model_dir'], 
                f'experiment3_trial_{trial.number}.pth'
            )
            save_model(model, optimizer, num_epochs, loss.item(), val_acc, save_path)
        
        return val_acc
    
    # Create Optuna study
    study = optuna.create_study(direction="maximize")
    print(f"Starting hyperparameter tuning with Optuna ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials)
    
    # Print best trial information
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (accuracy): {best_trial.value:.4f}")
    print(f"  Hyperparameters: {best_trial.params}")
    
    # Plot parameter importance if configured
    if exp_config.get('save_plots', False) and n_trials > 1:
        try:
            # Create parameter importance plot
            param_importance = optuna.visualization.plot_param_importances(study)
            
            # Save as image
            save_path = os.path.join(config['general']['log_dir'], 'experiment3_param_importance.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Convert to matplotlib figure and save
            fig = plt.figure(figsize=(10, 6))
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())
            values = list(importances.values())
            
            plt.barh(params, values)
            plt.xlabel('Importance')
            plt.title('Hyperparameter Importance')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"Parameter importance plot saved to {save_path}")
        except Exception as e:
            print(f"Could not create parameter importance plot: {e}")
    
    print("=== Experiment 3 Completed ===\n")
    
    return study


def run_all_experiments(config_path='config/acm_experiments.yaml', test_run=False):
    """
    Run all three experiments.
    
    Args:
        config_path (str): Path to the configuration file.
        test_run (bool): Whether this is a test run with reduced dataset size, epochs, and trials.
        
    Returns:
        tuple: Results from all experiments.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories if they don't exist
    os.makedirs(config['general']['log_dir'], exist_ok=True)
    os.makedirs(config['general']['model_dir'], exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])
    
    # Run experiments
    exp1_results = experiment1(config, test_run)
    exp2_results = experiment2(config, test_run)
    exp3_results = experiment3(config, test_run)
    
    return exp1_results, exp2_results, exp3_results


if __name__ == "__main__":
    # Run all experiments with test settings
    run_all_experiments(test_run=True)
