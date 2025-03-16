import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_cifar10(model, test_loader, device='cuda'):
    """
    Evaluate a model on CIFAR-10 test dataset.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        tuple: (accuracy, loss, predictions, targets)
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss, all_preds, all_targets

def evaluate_ptb(model, test_loader, device='cuda'):
    """
    Evaluate a language model on PTB test dataset.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        tuple: (perplexity, loss)
    """
    model = model.to(device)
    model.eval()
    
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    hidden = model.init_hidden(test_loader.batch_size, device)
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            hidden = tuple([h.detach() for h in hidden])
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss

def plot_training_history(history, title, save_path=None):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary containing training history
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'] if 'test_loss' in history else history['valid_loss'], 
             label='Test/Valid')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy or perplexity
    plt.subplot(2, 2, 2)
    if 'train_acc' in history:
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['test_acc'], label='Test')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
    else:
        plt.plot(history['train_ppl'], label='Train')
        plt.plot(history['valid_ppl'], label='Valid')
        plt.title('Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
    plt.legend()
    
    # Plot epoch times
    plt.subplot(2, 2, 3)
    plt.plot(history['epoch_times'])
    plt.title('Epoch Times')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    
    # Add title and adjust layout
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_optimizer_comparison(histories, metric, title, save_path=None):
    """
    Plot comparison of different optimizers.
    
    Args:
        histories: Dictionary of training histories for different optimizers
        metric: Metric to plot ('loss', 'acc', 'ppl')
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Map metric to actual keys in history
    metric_map = {
        'loss': 'train_loss',
        'acc': 'train_acc',
        'ppl': 'train_ppl',
        'val_loss': 'valid_loss',
        'val_acc': 'test_acc',
        'val_ppl': 'valid_ppl'
    }
    
    metric_key = metric_map.get(metric, metric)
    
    # Plot metric for each optimizer
    for optimizer_name, history in histories.items():
        if metric_key in history:
            plt.plot(history[metric_key], label=optimizer_name)
    
    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'{title} - {metric.capitalize()}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
