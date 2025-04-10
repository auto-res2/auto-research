"""
Visualization utilities for the DITTO-GSD experiments.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_figure(fig, filename):
    """
    Save figure as a high-quality PDF.
    
    Args:
        fig: The matplotlib figure.
        filename: The output filename (without extension).
    """
    ensure_directory('logs')
    fig.savefig(f'logs/{filename}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_loss_curve(epochs, losses, labels, title, filename):
    """
    Plot and save loss curves.
    
    Args:
        epochs: List of epoch numbers.
        losses: List of lists containing loss values for each model.
        labels: List of model labels.
        title: Plot title.
        filename: Output filename.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, loss in enumerate(losses):
        ax.plot(epochs, loss, marker='o', label=labels[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_figure(fig, filename)

def plot_degradation_comparison(conditions, baseline_scores, gsd_scores, title, filename):
    """
    Plot and save degradation comparison.
    
    Args:
        conditions: List of degradation conditions.
        baseline_scores: Scores for the baseline model.
        gsd_scores: Scores for the GSD model.
        title: Plot title.
        filename: Output filename.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))
    width = 0.35
    
    ax.bar(x - width/2, baseline_scores, width, label='Baseline DITTO')
    ax.bar(x + width/2, gsd_scores, width, label='DITTO-GSD')
    
    ax.set_xlabel('Degradation Condition')
    ax.set_ylabel('Reconstruction Quality')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, filename)
