"""
ACSC Data Preprocessing Module

This module handles the creation of synthetic test data for the ACSC experiments.
"""

import torch
import numpy as np

def create_synthetic_image(height=256, width=256):
    """
    Create a synthetic grayscale image with a horizontal and vertical gradient.
    
    Args:
      height: Height of the image
      width: Width of the image
      
    Returns:
      image_tensor: PyTorch tensor of shape [1, 1, H, W]
    """
    h_gradient = np.tile(np.linspace(0, 1, width), (height, 1)).astype(np.float32)
    
    v_gradient = np.tile(np.linspace(0, 1, height).reshape(height, 1), (1, width))
    
    gradient = (h_gradient + v_gradient) / 2.0
    
    image_tensor = torch.tensor(gradient).unsqueeze(0).unsqueeze(0)
    
    return image_tensor

def prepare_datasets(config):
    """
    Prepare datasets for experiments based on the provided configuration.
    
    Args:
      config: Configuration dictionary containing dataset parameters
      
    Returns:
      datasets: Dictionary containing prepared datasets
    """
    datasets = {}
    
    for res in config.get('resolutions', [128, 256, 512]):
        datasets[f'synthetic_{res}x{res}'] = create_synthetic_image(res, res)
    
    return datasets

def preprocess_data():
    """
    Main preprocessing function that prepares all data for experiments.
    
    Returns:
      preprocessed_data: Dictionary containing all preprocessed data
    """
    config = {
        'resolutions': [128, 256, 512],
    }
    
    datasets = prepare_datasets(config)
    
    return {
        'datasets': datasets,
        'config': config
    }

if __name__ == "__main__":
    data = preprocess_data()
    print(f"Preprocessed datasets: {list(data['datasets'].keys())}")
    print(f"Image shape for 256x256: {data['datasets']['synthetic_256x256'].shape}")
