"""
ClusterCloak: Visualization Utilities

This module contains functions for visualizing results of ClusterCloak experiments.
It includes functions for plotting t-SNE visualizations and training curves.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def plot_tsne(features, labels=None, title='t-SNE Projection', filename=None, colors=None):
    """
    Create t-SNE visualization of feature embeddings.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Optional labels for coloring points
        title: Plot title
        filename: If provided, saves the plot to this filename (PDF format)
        colors: Optional color specification
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_proj = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors if colors is not None else 'blue')
    plt.title(title)
    
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved t-SNE plot as: {filename}")
    
    plt.close()
