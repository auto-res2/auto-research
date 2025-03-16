"""
Model evaluation module for ACM optimizer experiments.

This module implements evaluation functions for:
1. Synthetic function optimization results
2. CIFAR-10 image classification model evaluation
3. Ablation study comparisons
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_synthetic_results(results, func_name, save_dir='./paper'):
    """
    Evaluate and visualize results from synthetic function optimization.
    
    Args:
        results (dict): Dictionary with optimizer names as keys and results as values
        func_name (str): Name of the function ('rosenbrock' or 'ill_conditioned')
        save_dir (str): Directory to save plots
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n======== Evaluating {func_name} optimization results ========")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare plot
    plt.figure(figsize=(12, 10))
    
    # Plot convergence curves (loss vs. iterations)
    plt.subplot(2, 1, 1)
    for name, res in results.items():
        losses = res['losses']
        plt.semilogy(losses, label=name)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Convergence on {func_name} function')
    plt.legend()
    plt.grid(True)
    
    # Plot trajectories on contour plot
    plt.subplot(2, 1, 2)
    
    if func_name == 'rosenbrock':
        # Rosenbrock contour
        x = np.linspace(-2, 2, 400)
        y = np.linspace(-1, 3, 400)
        X, Y = np.meshgrid(x, y)
        Z = (1 - X)**2 + 100 * (Y - X**2)**2
        levels = np.logspace(-1, 3, 20)
        plt.contour(X, Y, Z, levels=levels, cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectory on Rosenbrock Function')
        
    elif func_name == 'ill_conditioned':
        # Ill-conditioned quadratic contour
        x_vals = np.linspace(-6, 6, 400)
        y_vals = np.linspace(-6, 6, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = 0.5*(X**2 + 100*Y**2)
        levels = np.logspace(0, 3, 20)
        plt.contour(X, Y, Z, levels=levels, cmap='plasma')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectory on Ill-conditioned Quadratic Function')
    
    # Plot trajectories
    for name, res in results.items():
        traj = res['trajectory']
        plt.plot(traj[:, 0], traj[:, 1], label=name, linewidth=2)
    
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{func_name}_optimization.png'))
    print(f"Plot saved to {os.path.join(save_dir, f'{func_name}_optimization.png')}")
    
    # Calculate metrics
    metrics = {}
    for name, res in results.items():
        losses = res['losses']
        metrics[name] = {
            'final_loss': losses[-1],
            'iterations': len(losses),
            'convergence_rate': np.mean(np.diff(np.log(losses[1:]))) if len(losses) > 1 else 0
        }
        print(f"{name}: Final loss = {losses[-1]:.4e}, Iterations = {len(losses)}")
    
    return metrics

def evaluate_cifar10_model(model, test_loader, device=None, save_dir='./paper'):
    """
    Evaluate a trained model on CIFAR-10 test set.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use for evaluation
        save_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n======== Evaluating CIFAR-10 model ========")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Initialize variables
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # Evaluate without gradient calculation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save confusion matrix plot
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Generate classification report
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(str(report))
    print(f"Classification report saved to {report_path}")
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics

def compare_optimizers(histories, save_dir='./paper'):
    """
    Compare different optimizers based on training histories.
    
    Args:
        histories (dict): Dictionary with optimizer names as keys and training histories as values
        save_dir (str): Directory to save comparison plots
        
    Returns:
        dict: Comparison metrics
    """
    print("\n======== Comparing Optimizers ========")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(15, 5))
    
    # Training loss
    plt.subplot(1, 3, 1)
    for name, history in histories.items():
        plt.plot(history['train_loss'], label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Validation loss
    plt.subplot(1, 3, 2)
    for name, history in histories.items():
        plt.plot(history['val_loss'], label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Validation accuracy
    plt.subplot(1, 3, 3)
    for name, history in histories.items():
        plt.plot(history['val_acc'], label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    # Save comparison plot
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'optimizer_comparison.png')
    plt.savefig(comparison_path)
    print(f"Comparison plot saved to {comparison_path}")
    
    # Calculate comparison metrics
    metrics = {}
    for name, history in histories.items():
        metrics[name] = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_acc': history['val_acc'][-1],
            'avg_epoch_time': np.mean(history['epoch_times']),
            'convergence_rate': np.mean(np.diff(history['val_loss']))
        }
        print(f"{name}: Final val acc = {history['val_acc'][-1]:.4f}, "
              f"Final val loss = {history['val_loss'][-1]:.4f}, "
              f"Avg epoch time = {np.mean(history['epoch_times']):.2f}s")
    
    return metrics

def evaluate_ablation_results(results, save_dir='./paper'):
    """
    Evaluate and visualize results from ablation studies.
    
    Args:
        results (dict): Dictionary with optimizer variants as keys and results as values
        save_dir (str): Directory to save plots
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n======== Evaluating Ablation Study Results ========")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot convergence curves
    plt.figure(figsize=(12, 6))
    
    for name, res in results.items():
        losses = res['losses']
        plt.semilogy(losses, label=name)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.title('Convergence Comparison of ACM Variants')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    ablation_path = os.path.join(save_dir, 'ablation_study.png')
    plt.savefig(ablation_path)
    print(f"Ablation study plot saved to {ablation_path}")
    
    # Calculate metrics
    metrics = {}
    for name, res in results.items():
        losses = res['losses']
        metrics[name] = {
            'final_loss': losses[-1],
            'iterations': len(losses),
            'convergence_rate': np.mean(np.diff(np.log(losses[1:]))) if len(losses) > 1 else 0
        }
        print(f"{name}: Final loss = {losses[-1]:.4e}, Iterations = {len(losses)}")
    
    return metrics
