"""
Visualization utilities for PurifyCov++ experiments.
"""

import matplotlib.pyplot as plt
import numpy as np

def save_boxplot(data_list, labels, title, ylabel, filename):
    """
    Create and save a boxplot.
    
    Args:
        data_list: List of data to plot
        labels: Labels for x-axis
        title: Plot title
        ylabel: Label for y-axis
        filename: Output filename
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_list)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_line_plot(x_data, y_data_list, labels, title, xlabel, ylabel, filename):
    """
    Create and save a line plot.
    
    Args:
        x_data: Data for x-axis
        y_data_list: List of data for y-axis
        labels: Labels for legend
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        filename: Output filename
    """
    plt.figure(figsize=(10, 6))
    for i, y_data in enumerate(y_data_list):
        plt.plot(x_data, y_data, marker='o' if i == 0 else 's', label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
