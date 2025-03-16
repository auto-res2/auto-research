"""
Evaluation module for ACM optimizer experiments.

This module implements the evaluation procedures for comparing the Adaptive Curvature
Momentum (ACM) optimizer with other standard optimizers.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.train import get_model


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate a model on the test set.
    
    Args:
        model (nn.Module): PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        test_loss, test_acc: Test loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    test_loss = running_loss / total
    test_acc = 100. * correct / total
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return test_loss, test_acc, all_preds, all_targets


def evaluate_quadratic(model, x_data, y_data, criterion, device):
    """
    Evaluate a model on the quadratic function data.
    
    Args:
        model (nn.Module): PyTorch model
        x_data (Tensor): Input data
        y_data (Tensor): Target data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        loss: Test loss
    """
    model.eval()
    
    # Move data to device
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(x_data)
        loss = criterion(outputs, y_data.unsqueeze(1))
    
    return loss.item()


def plot_training_curves(results, save_dir='./logs'):
    """
    Plot training curves for different optimizers.
    
    Args:
        results (dict): Dictionary containing training results for different optimizers
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    for optimizer_name, history in results.items():
        plt.plot(history['train_loss'], label=f'{optimizer_name}')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()
    
    # Plot validation accuracy
    plt.figure(figsize=(10, 6))
    for optimizer_name, history in results.items():
        plt.plot(history['val_acc'], label=f'{optimizer_name}')
    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'validation_accuracy.png'))
    plt.close()


def plot_confusion_matrix(all_preds, all_targets, class_names, save_dir='./logs'):
    """
    Plot confusion matrix.
    
    Args:
        all_preds (array): Predicted labels
        all_targets (array): True labels
        class_names (list): List of class names
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()


def plot_quadratic_results(results, save_dir='./logs'):
    """
    Plot results for the quadratic function experiment.
    
    Args:
        results (dict): Dictionary containing training results for different optimizers
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for optimizer_name, history in results.items():
        plt.plot(history['loss'], label=f'{optimizer_name}')
    plt.title('Loss vs. Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'quadratic_loss.png'))
    plt.close()
    
    # Plot curvature estimates for ACM
    if 'acm' in results:
        plt.figure(figsize=(10, 6))
        curvature_data = results['acm']['curvature']
        for i, curve in enumerate(zip(*curvature_data)):
            plt.plot(curve, label=f'Parameter {i+1}')
        plt.title('ACM: Curvature Estimates vs. Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Curvature Estimate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'acm_curvature.png'))
        plt.close()


def plot_hyperparameter_heatmap(results, save_dir='./logs'):
    """
    Plot heatmap for hyperparameter search results.
    
    Args:
        results (dict): Dictionary containing hyperparameter search results
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert results to DataFrame
    data = []
    for params, val_acc in results.items():
        lr, curvature_coef = params
        data.append({'lr': lr, 'curvature_coef': curvature_coef, 'val_acc': val_acc})
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot_table = pd.pivot_table(df, values='val_acc', index='curvature_coef', columns='lr')
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Validation Accuracy Heatmap (ACM)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Curvature Coefficient')
    plt.savefig(os.path.join(save_dir, 'hyperparameter_heatmap.png'))
    plt.close()


def evaluate_experiment(config, results, data):
    """
    Evaluate experiment results.
    
    Args:
        config (dict): Configuration dictionary
        results (dict): Dictionary containing training results
        data (dict): Dictionary containing data loaders or datasets
        
    Returns:
        eval_results: Dictionary containing evaluation results
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    experiment_type = config.get('experiment_type', 'cifar10')
    model_name = config.get('model_name', 'resnet18')
    optimizer_names = config.get('optimizers', ['sgd', 'adam', 'adabelief', 'acm'])
    save_dir = config.get('save_dir', './logs')
    
    eval_results = {}
    
    if experiment_type == 'cifar10':
        criterion = nn.CrossEntropyLoss()
        test_loader = data['test_loader']
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Plot training curves
        plot_training_curves(results, save_dir)
        
        # Evaluate each model
        for optimizer_name in optimizer_names:
            print(f'\nEvaluating model trained with {optimizer_name}')
            
            # Load best model
            model = get_model(model_name, config)
            model_path = os.path.join(config.get('save_dir', './models'), 
                                     f'{model_name}_{optimizer_name}_best.pth')
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                model = model.to(device)
                
                # Evaluate model
                test_loss, test_acc, all_preds, all_targets = evaluate_model(
                    model, test_loader, criterion, device
                )
                
                print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
                
                # Generate classification report
                report = classification_report(all_targets, all_preds, 
                                              target_names=class_names, 
                                              output_dict=True)
                
                # Plot confusion matrix
                plot_confusion_matrix(all_preds, all_targets, class_names, 
                                     os.path.join(save_dir, optimizer_name))
                
                # Store results
                eval_results[optimizer_name] = {
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'classification_report': report
                }
            else:
                print(f'Model file not found: {model_path}')
        
        # Compare optimizers
        if eval_results:
            # Create comparison table
            comparison = {
                'Optimizer': [],
                'Test Accuracy (%)': [],
                'Test Loss': []
            }
            
            for optimizer_name, result in eval_results.items():
                comparison['Optimizer'].append(optimizer_name)
                comparison['Test Accuracy (%)'].append(result['test_acc'])
                comparison['Test Loss'].append(result['test_loss'])
            
            # Print comparison table
            df = pd.DataFrame(comparison)
            print('\nOptimizer Comparison:')
            print(df.to_string(index=False))
            
            # Save comparison to CSV
            df.to_csv(os.path.join(save_dir, 'optimizer_comparison.csv'), index=False)
    
    elif experiment_type == 'quadratic':
        # Plot quadratic results
        plot_quadratic_results(results, save_dir)
    
    elif experiment_type == 'hyperparameter_search':
        # Plot hyperparameter heatmap
        plot_hyperparameter_heatmap(results, save_dir)
    
    return eval_results
