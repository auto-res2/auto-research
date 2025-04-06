"""
ClusterCloak: Transformation Utilities

This module contains transformation functions used in ClusterCloak experiments.
It includes functions for applying poisoning perturbations and image transformations.
"""

import torch
import numpy as np
import cv2

def apply_clustercloak(image, epsilon=0.05, clustering_bias=0.02):
    """
    Apply ClusterCloak perturbation to images.
    
    Args:
        image: Tensor image(s) to be perturbed
        epsilon: Strength of random noise
        clustering_bias: Strength of clustering bias
        
    Returns:
        Perturbed image tensor
    """
    noise = torch.randn_like(image) * epsilon
    cluster_component = clustering_bias * torch.sign(image)
    return image + noise + cluster_component

def apply_metacloak(image, epsilon=0.05):
    """
    Apply MetaCloak perturbation to images.
    
    Args:
        image: Tensor image(s) to be perturbed
        epsilon: Strength of random noise
        
    Returns:
        Perturbed image tensor
    """
    noise = torch.randn_like(image) * epsilon
    return image + noise
