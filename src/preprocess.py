"""
Data preprocessing for MML-BO experiments.
"""
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def create_synthetic_functions():
    """
    Creates synthetic functions for optimization tasks.
    
    Returns:
        dict: Dictionary containing synthetic functions for optimization.
    """
    def synthetic_func_quadratic(x, noise_std=0.1):
        """
        Quadratic function with noise.
        f(x) = (x - 2)^2 + noise
        
        Args:
            x (np.ndarray): Input values
            noise_std (float): Standard deviation of Gaussian noise
            
        Returns:
            np.ndarray: Function values with added noise
        """
        noise = np.random.normal(0, noise_std, size=x.shape)
        return (x - 2)**2 + noise

    def synthetic_func_sinusoidal(x, noise_std=1.0):
        """
        Sinusoidal function with noise.
        f(x) = sin(3x) + 0.5x + noise
        
        Args:
            x (np.ndarray): Input values
            noise_std (float): Standard deviation of Gaussian noise
            
        Returns:
            np.ndarray: Function values with added noise
        """
        noise = np.random.normal(0, noise_std, size=x.shape)
        return np.sin(3 * x) + 0.5 * x + noise
    
    return {
        "quadratic": synthetic_func_quadratic,
        "sinusoidal": synthetic_func_sinusoidal
    }

def create_meta_learning_data(n_samples=100, feature_dim=10):
    """
    Creates synthetic data for meta-learning experiments.
    
    Args:
        n_samples (int): Number of data samples to generate
        feature_dim (int): Dimension of the feature vectors
        
    Returns:
        tuple: (data, target) where data is the feature tensor and target is the corresponding labels
    """
    data = torch.randn(n_samples, feature_dim)
    target = torch.randn(n_samples, 1)
    
    os.makedirs('data/meta_learning', exist_ok=True)
    torch.save(data, 'data/meta_learning/features.pt')
    torch.save(target, 'data/meta_learning/targets.pt')
    
    return data, target
