"""
Utility functions for model evaluation.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from .data_utils import augment_point_cloud

def evaluate_model(model, dataset, num_augmentations=3, device='cuda'):
    """
    Evaluate model robustness by applying augmentations to point clouds.
    
    For each sample, the model's prediction is obtained for several augmentations.
    Majority vote is then used to compute accuracy.
    
    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        num_augmentations: Number of augmentations to apply to each sample
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        accuracy: Accuracy of the model on the augmented dataset
    """
    model.to(device)
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in DataLoader(dataset, batch_size=8, shuffle=False):
            inputs_np = inputs.numpy()
            B = inputs_np.shape[0]
            
            aug_preds = []
            for _ in range(num_augmentations):
                batch_aug = np.stack([augment_point_cloud(pc) for pc in inputs_np])
                batch_tensor = torch.tensor(batch_aug, dtype=torch.float32).to(device)
                outputs = model(batch_tensor)
                preds = outputs.argmax(dim=1).cpu().numpy()
                aug_preds.append(preds)
                
            aug_preds = np.array(aug_preds)  # (num_augmentations, B)
            
            majority_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=aug_preds)
            
            labels_np = labels.numpy()
            correct = (majority_preds == labels_np).sum()
            
            total_correct += correct
            total_samples += B
    
    accuracy = total_correct / total_samples
    return accuracy

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, filename='training_curves.pdf'):
    """
    Plot training curves and save as PDF.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        filename: Name of the file to save the plot to
    """
    os.makedirs('logs', exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('logs', filename), format='pdf', dpi=300)
    plt.close()
    
    print(f"Saved training curves to logs/{filename}")

def plot_comparison_bar(accuracies, labels, title, filename):
    """
    Plot a comparison bar chart and save as PDF.
    
    Args:
        accuracies: List of accuracies to compare
        labels: List of labels for the bars
        title: Title of the plot
        filename: Name of the file to save the plot to
    """
    os.makedirs('logs', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=['blue', 'green', 'red', 'orange', 'purple'])
    
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join('logs', filename), format='pdf', dpi=300)
    plt.close()
    
    print(f"Saved comparison bar chart to logs/{filename}")
