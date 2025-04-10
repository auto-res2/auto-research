"""
Data preprocessing module for DITTO-GSD experiments.
"""
import numpy as np
import torch

def add_gaussian_noise(pc, sigma=0.02):
    """
    Add Gaussian noise to the point cloud.
    
    Args:
        pc: Input point cloud.
        sigma: Standard deviation of Gaussian noise.
        
    Returns:
        Noisy point cloud.
    """
    if isinstance(pc, torch.Tensor):
        noise = torch.randn_like(pc) * sigma
        return pc + noise
    else:
        noise = np.random.normal(0, sigma, pc.shape).astype(np.float32)
        return pc + noise

def downsample(pc, keep_ratio=0.5):
    """
    Downsample the point cloud.
    
    Args:
        pc: Input point cloud.
        keep_ratio: Ratio of points to keep.
        
    Returns:
        Downsampled point cloud.
    """
    if isinstance(pc, torch.Tensor):
        num_points = int(pc.shape[0] * keep_ratio)
        indices = torch.randperm(pc.shape[0])[:num_points]
        return pc[indices]
    else:
        num_points = int(pc.shape[0] * keep_ratio)
        indices = np.random.choice(pc.shape[0], num_points, replace=False)
        return pc[indices]

def generate_degraded_versions(pc, noise_levels, sparsity_levels):
    """
    Generate dictionary of degraded point clouds.
    
    Args:
        pc: Input point cloud.
        noise_levels: List of noise levels (sigma).
        sparsity_levels: List of sparsity levels (keep_ratio).
        
    Returns:
        Dictionary of degraded point clouds.
    """
    degraded_versions = {}
    for sigma in noise_levels:
        noisy_pc = add_gaussian_noise(pc, sigma)
        degraded_versions[f"noise_{sigma}"] = noisy_pc
    for ratio in sparsity_levels:
        sparse_pc = downsample(pc, ratio)
        degraded_versions[f"sparsity_{int(ratio*100)}%"] = sparse_pc
    return degraded_versions

def preprocess_data(dataset, batch_size=4, shuffle=True):
    """
    Preprocess the dataset and create a dataloader.
    
    Args:
        dataset: Input dataset.
        batch_size: Batch size for dataloader.
        shuffle: Whether to shuffle the dataloader.
        
    Returns:
        Dataloader for the preprocessed dataset.
    """
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
