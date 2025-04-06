"""
ClusterCloak: Metrics Utilities

This module contains functions for evaluating clustering quality and model performance.
It includes metrics for clustering evaluation and identity preservation.
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def compute_clustering_metrics(features, labels):
    """
    Compute clustering quality metrics.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Cluster labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    try:
        metrics['silhouette'] = silhouette_score(features, labels)
    except Exception as e:
        metrics['silhouette'] = -1
        print(f"Error computing silhouette score: {e}")
    
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
    except Exception as e:
        metrics['davies_bouldin'] = -1
        print(f"Error computing Davies-Bouldin score: {e}")
        
    return metrics
