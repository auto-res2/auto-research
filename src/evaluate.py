"""
Model evaluation module for the auto-research project.
Contains functions for evaluating models and visualizing results.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.preprocess import get_device, synthetic_function

def evaluate_cifar10_model(model, testloader, device=None):
    """
    Evaluate a trained model on the CIFAR-10 test set.
    
    Args:
        model (nn.Module): Trained model
        testloader (DataLoader): Test data loader
        device (torch.device): Device to use for evaluation
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if device is None:
        device = get_device()
    
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Print results
    print(f"CIFAR-10 Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics

def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def plot_training_history(train_losses, test_accuracies, optimizer_name, save_path=None):
    """
    Plot training loss and test accuracy.
    
    Args:
        train_losses (list): Training losses
        test_accuracies (list): Test accuracies
        optimizer_name (str): Name of the optimizer used
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-')
    plt.title(f'Training Loss ({optimizer_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, 'r-')
    plt.title(f'Test Accuracy ({optimizer_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.close()

def visualize_synthetic_trajectories(trajectory_acm, trajectory_adam, save_path=None):
    """
    Visualize optimizer trajectories on the synthetic function landscape.
    
    Args:
        trajectory_acm (numpy.ndarray): Trajectory of the ACM optimizer
        trajectory_adam (numpy.ndarray): Trajectory of the Adam optimizer
        save_path (str): Path to save the plot
    """
    # Create a grid for contour plotting
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calculate function values on the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xy = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
            Z[i, j] = synthetic_function(xy).item()
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot contours
    contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot trajectories
    plt.plot(trajectory_acm[:, 0], trajectory_acm[:, 1], 'ro-', label='ACM', linewidth=2, markersize=4)
    plt.plot(trajectory_adam[:, 0], trajectory_adam[:, 1], 'bo-', label='Adam', linewidth=2, markersize=4)
    
    # Mark start and end points
    plt.plot(trajectory_acm[0, 0], trajectory_acm[0, 1], 'r*', markersize=10, label='Start')
    plt.plot(trajectory_acm[-1, 0], trajectory_acm[-1, 1], 'rx', markersize=10, label='ACM End')
    plt.plot(trajectory_adam[-1, 0], trajectory_adam[-1, 1], 'bx', markersize=10, label='Adam End')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimizer Trajectories on Synthetic Landscape')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Trajectory visualization saved to {save_path}")
    
    plt.close()

def plot_two_moons_decision_boundary(model, X_val, y_val, save_path=None):
    """
    Plot the decision boundary of a trained model on the two-moons dataset.
    
    Args:
        model (nn.Module): Trained model
        X_val (torch.Tensor): Validation features
        y_val (torch.Tensor): Validation labels
        save_path (str): Path to save the plot
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create a grid for visualization
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict class for each point in the grid
    with torch.no_grad():
        grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        Z = model(grid_points)
        Z = torch.argmax(Z, dim=1).numpy()
    
    # Reshape the predictions
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    
    # Plot the validation data
    scatter = plt.scatter(X_val[:, 0].numpy(), X_val[:, 1].numpy(), 
                         c=y_val.numpy(), cmap=plt.cm.RdBu, 
                         edgecolors='k', s=40)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Two-Moons Decision Boundary')
    plt.colorbar(scatter)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Decision boundary plot saved to {save_path}")
    
    plt.close()

def compare_ablation_results(results, save_path=None):
    """
    Compare and visualize results from the ablation study.
    
    Args:
        results (dict): Dictionary containing results from different hyperparameter settings
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    for key, data in results.items():
        plt.plot(data['losses'], label=f"{key} (acc={data['accuracy']:.2f})")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Ablation Study: Loss Curves (Two-Moons)')
    plt.legend()
    plt.grid(True)
    
    # Plot final accuracies as a heatmap
    plt.subplot(1, 2, 2)
    
    # Extract beta and curvature_scale values
    beta_values = sorted(list(set([float(key.split('_')[1]) for key in results.keys()])))
    cs_values = sorted(list(set([float(key.split('_')[3]) for key in results.keys()])))
    
    # Create accuracy matrix
    accuracy_matrix = np.zeros((len(beta_values), len(cs_values)))
    for i, beta in enumerate(beta_values):
        for j, cs in enumerate(cs_values):
            key = f"beta_{beta}_cs_{cs}"
            accuracy_matrix[i, j] = results[key]['accuracy']
    
    # Plot heatmap
    im = plt.imshow(accuracy_matrix, interpolation='nearest', cmap=plt.cm.viridis)
    plt.colorbar(im, label='Validation Accuracy')
    
    # Set ticks and labels
    plt.xticks(np.arange(len(cs_values)), [str(cs) for cs in cs_values])
    plt.yticks(np.arange(len(beta_values)), [str(beta) for beta in beta_values])
    
    plt.xlabel('Curvature Scale')
    plt.ylabel('Beta')
    plt.title('Validation Accuracy by Hyperparameters')
    
    # Add text annotations
    for i in range(len(beta_values)):
        for j in range(len(cs_values)):
            plt.text(j, i, f"{accuracy_matrix[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if accuracy_matrix[i, j] < 0.85 else "black")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Ablation study comparison saved to {save_path}")
    
    plt.close()
