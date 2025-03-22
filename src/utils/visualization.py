"""
Utility functions for visualizing experiment results.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import os


def plot_loss_curves(results, title, xlabel="Epoch", ylabel="Loss", save_path=None):
    """
    Plot loss curves for different configurations.
    
    Args:
        results: Dictionary mapping configuration names to loss histories
        title: Title of the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the figure, if provided
    """
    plt.figure(figsize=(10, 6))
    for config, losses in results.items():
        plt.plot(losses, label=f"{config}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.close()


def visualize_samples(samples, title="Generated Samples", nrow=8, save_path=None):
    """
    Visualize a batch of samples.
    
    Args:
        samples: Tensor of shape [B, C, H, W]
        title: Title of the plot
        nrow: Number of images per row
        save_path: Path to save the figure, if provided
    """
    # Ensure the samples are in range [0, 1]
    if samples.min() < 0 or samples.max() > 1:
        samples = (samples - samples.min()) / (samples.max() - samples.min())
    
    # Create a grid of images
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    grid = grid.cpu().numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_schedules(schedules, labels, title="Diffusion Schedules", save_path=None):
    """
    Plot different diffusion schedules.
    
    Args:
        schedules: List of schedules to plot
        labels: Labels for each schedule
        title: Title of the plot
        save_path: Path to save the figure, if provided
    """
    plt.figure(figsize=(10, 6))
    for i, schedule in enumerate(schedules):
        plt.plot(schedule, label=labels[i])
    plt.xlabel("Diffusion Step")
    plt.ylabel("Noise Level")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.close()
