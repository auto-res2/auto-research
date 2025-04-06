"""
Visualization utilities for ProtoSurvPath experiments.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_directory_check(directory='logs'):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_training_loss(loss_history, title="Training Loss", condition="", save_dir="logs"):
    """
    Plot training loss curve and save as PDF.
    
    Args:
        loss_history: List of loss values per epoch
        title: Plot title
        condition: Model condition/name for filename
        save_dir: Directory to save the plot
    """
    save_directory_check(save_dir)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(loss_history)+1), y=loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    filename = f"{save_dir}/training_loss_{condition}.pdf" if condition else f"{save_dir}/training_loss.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()

def plot_metric(metric_history, title="Concordance Index", condition="", save_dir="logs"):
    """
    Plot evaluation metric curve and save as PDF.
    
    Args:
        metric_history: List of metric values per epoch
        title: Plot title
        condition: Model condition/name for filename
        save_dir: Directory to save the plot
    """
    save_directory_check(save_dir)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(metric_history)+1), y=metric_history)
    plt.xlabel("Epoch")
    plt.ylabel("c-index")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    filename = f"{save_dir}/cindex_{condition}.pdf" if condition else f"{save_dir}/cindex.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()

def plot_integrated_gradients(attributions, title="Integrated Gradients Attribution", condition="", save_dir="logs"):
    """
    Plot integrated gradients attributions and save as PDF.
    
    Args:
        attributions: Array of attribution values
        title: Plot title
        condition: Model condition/name for filename
        save_dir: Directory to save the plot
    """
    save_directory_check(save_dir)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(attributions)), attributions)
    plt.xlabel("Gene feature index")
    plt.ylabel("Attribution")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    filename = f"{save_dir}/interpretability_{condition}.pdf" if condition else f"{save_dir}/interpretability.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()

def plot_confusion_matrix(confusion_matrix, class_names, title="Confusion Matrix", condition="", save_dir="logs"):
    """
    Plot confusion matrix and save as PDF.
    
    Args:
        confusion_matrix: Numpy array of confusion matrix values
        class_names: List of class names
        title: Plot title
        condition: Model condition/name for filename
        save_dir: Directory to save the plot
    """
    save_directory_check(save_dir)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    
    filename = f"{save_dir}/confusion_matrix_{condition}.pdf" if condition else f"{save_dir}/confusion_matrix.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()

def plot_prototype_visualization(prototypes, title="Prototype Visualization", condition="", save_dir="logs"):
    """
    Visualize learned prototypes and save as PDF.
    
    Args:
        prototypes: Array of prototype values
        title: Plot title
        condition: Model condition/name for filename
        save_dir: Directory to save the plot
    """
    save_directory_check(save_dir)
    num_prototypes, dim = prototypes.shape
    
    plt.figure(figsize=(12, 8))
    for i in range(num_prototypes):
        plt.subplot(num_prototypes, 1, i+1)
        plt.bar(range(dim), prototypes[i])
        plt.title(f"Prototype {i+1}")
        
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    filename = f"{save_dir}/prototype_{condition}.pdf" if condition else f"{save_dir}/prototype.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()
