"""Visualization utilities for GraphDiffLayout experiments."""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def ensure_directory(path):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_figure(fig, filename, directory='logs', format='pdf', dpi=300):
    """
    Save figure in high-quality format suitable for academic papers.
    
    Args:
        fig: matplotlib figure object
        filename: name of the file without extension
        directory: directory to save the figure
        format: file format (pdf, png, etc.)
        dpi: resolution in dots per inch
    """
    ensure_directory(directory)
    filepath = os.path.join(directory, f"{filename}.{format}")
    fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: {filepath}")
    
    return filepath

def visualize_comparison(img1, img2, title1, title2, suptitle, filename, 
                         directory='logs', format='pdf', dpi=300):
    """
    Create and save a side-by-side comparison of two images.
    
    Args:
        img1, img2: numpy array images (BGR format)
        title1, title2: titles for the two images
        suptitle: overall title for the figure
        filename: name of the file without extension
        directory: directory to save the figure
        format: file format (pdf, png, etc.)
        dpi: resolution in dots per inch
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title1)
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title2)
    axes[1].axis("off")
    
    plt.suptitle(suptitle)
    plt.tight_layout()
    
    filepath = save_figure(fig, filename, directory, format, dpi)
    plt.close(fig)
    
    return filepath
