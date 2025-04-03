"""
Data preprocessing module for ASMO-Solver experiments.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA

def generate_synthetic_trajectory(timesteps, latent_dim, seed=None):
    """
    Generate a synthetic trajectory for experiments.
    
    Args:
        timesteps (int): Number of time steps in the trajectory
        latent_dim (int): Dimensionality of the latent space
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        np.ndarray: Synthetic trajectory of shape (timesteps, latent_dim)
    """
    if seed is not None:
        np.random.seed(seed)
    
    trajectory = np.cumsum(np.random.randn(timesteps, latent_dim), axis=0)
    return trajectory

def static_projection(X, target_dim=2):
    """
    Project data to a fixed dimensionality using PCA.
    
    Args:
        X (np.ndarray): Data matrix of shape (n_samples, n_features)
        target_dim (int): Target dimensionality
        
    Returns:
        np.ndarray: Projected data of shape (n_samples, target_dim)
    """
    pca = PCA(n_components=target_dim)
    projected = pca.fit_transform(X)
    return projected

def dynamic_projection(X, min_dim=2, max_dim=5, threshold=0.1):
    """
    Dynamically project data to a lower-dimensional space.
    
    Args:
        X (np.ndarray): Data matrix of shape (n_samples, n_features)
        min_dim (int): Minimum dimensionality
        max_dim (int): Maximum dimensionality
        threshold (float): Threshold for determining effective dimensions
        
    Returns:
        tuple: (projected_data, effective_dims)
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    effective_dims = np.sum(S >= threshold * S[0])
    effective_dims = int(np.clip(effective_dims, min_dim, max_dim))
    
    dynamic_proj = Vt[:effective_dims, :]
    projected = np.dot(X_centered, dynamic_proj.T)
    
    return projected, effective_dims

def setup_device(device_str="cuda"):
    """
    Set up the device for PyTorch computations.
    
    Args:
        device_str (str): 'cuda' or 'cpu'
        
    Returns:
        torch.device: PyTorch device
    """
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device
