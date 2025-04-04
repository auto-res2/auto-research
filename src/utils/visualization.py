"""
Visualization utilities for the SAC-Seg experiments.
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union


def save_plot(plt_figure: Figure, filename: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure as a high-quality PDF.
    
    Args:
        plt_figure: The matplotlib figure to save
        filename: The output filename (must end with .pdf)
        dpi: The resolution in dots per inch
    """
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    plt_figure.savefig(filename, format='pdf', dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {filename}")


def plot_comparison(
    data_dict: Dict[str, List[float]],
    x_values: Optional[List[Union[int, float]]] = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    filename: str = 'comparison_plot.pdf'
) -> Figure:
    """
    Create and save a comparison plot for multiple data series.
    
    Args:
        data_dict: Dictionary with method names as keys and data lists as values
        x_values: Optional x-axis values, defaults to indices if not provided
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
        
    Returns:
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for label, data in data_dict.items():
        if x_values is not None:
            ax.plot(x_values[:len(data)], data, marker='o', label=label)
        else:
            ax.plot(data, marker='o', label=label)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    
    save_plot(fig, filename)
    return fig
