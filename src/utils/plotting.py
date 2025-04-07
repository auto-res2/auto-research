"""
Utility functions for creating and saving figures.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def save_figure(filename, dpi=300, directory="logs"):
    """
    Save the current matplotlib figure as a high-quality PDF.
    
    Args:
        filename (str): Name of the file (without extension)
        dpi (int): DPI for the figure
        directory (str): Directory to save the figure
        
    Returns:
        str: Full path to the saved figure
    """
    os.makedirs(directory, exist_ok=True)
    
    if not filename.endswith('.pdf'):
        filename = f"{filename}.pdf"
    
    filepath = os.path.join(directory, filename)
    
    plt.savefig(filepath, dpi=dpi, format="pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Figure saved as {filepath}")
    return filepath

def plot_comparison_bar(labels, values, ylabel, title, filename, color=['blue', 'green']):
    """
    Plot a bar chart comparing two values.
    
    Args:
        labels (list): Labels for the x-axis
        values (list): Values for the bars
        ylabel (str): Label for the y-axis
        title (str): Title for the plot
        filename (str): Name of the file to save
        color (list): Colors for the bars
        
    Returns:
        str: Full path to the saved figure
    """
    plt.figure()
    plt.bar(labels, values, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    
    return save_figure(filename)

def plot_line(x, y, xlabel, ylabel, title, filename, marker='o', linestyle='-'):
    """
    Plot a line chart.
    
    Args:
        x (list): Values for the x-axis
        y (list): Values for the y-axis
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        title (str): Title for the plot
        filename (str): Name of the file to save
        marker (str): Marker style
        linestyle (str): Line style
        
    Returns:
        str: Full path to the saved figure
    """
    plt.figure()
    plt.plot(x, y, marker=marker, linestyle=linestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    return save_figure(filename)

def plot_heatmap(data, title, filename, cmap='viridis'):
    """
    Plot a heatmap.
    
    Args:
        data (numpy.ndarray): 2D array of data
        title (str): Title for the plot
        filename (str): Name of the file to save
        cmap (str): Colormap to use
        
    Returns:
        str: Full path to the saved figure
    """
    plt.figure()
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    return save_figure(filename)
