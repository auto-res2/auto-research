"""
RapidAlign: Training module

This module implements the diffusion samplers for RapidAlign.
"""

import torch
import time
import numpy as np
from typing import List, Tuple, Optional

def generate_latent_vector(dim=16):
    """
    Generate a random latent vector.
    
    Args:
        dim (int): Dimension of the latent vector
        
    Returns:
        torch.Tensor: Random latent vector
    """
    return torch.randn(dim)

def ddim_sampler(latent, num_steps=50, noise_scale=0.1):
    """
    Standard DDIM sampler with simplified update rule.
    
    Args:
        latent (torch.Tensor): Initial latent vector
        num_steps (int): Number of sampling steps
        noise_scale (float): Scale of noise to add
        
    Returns:
        List[torch.Tensor]: Trajectory of latent vectors
    """
    trajectory = [latent.clone()]
    for step in range(num_steps):
        noise = noise_scale * torch.randn_like(latent)
        latent = latent - 0.1 * latent + noise  
        trajectory.append(latent.clone())
    return trajectory

def rapidalign_sampler(latent, num_steps=50, noise_scale=0.1):
    """
    RapidAlign sampler using adaptive Heun's method.
    
    Args:
        latent (torch.Tensor): Initial latent vector
        num_steps (int): Number of sampling steps
        noise_scale (float): Scale of noise to add
        
    Returns:
        List[torch.Tensor]: Trajectory of latent vectors
    """
    trajectory = [latent.clone()]
    dt = 0.1
    def f(x):
        return -x
    for step in range(num_steps):
        euler_est = latent + dt * f(latent) + noise_scale * torch.randn_like(latent)
        latent = latent + (dt/2) * (f(latent) + f(euler_est))
        trajectory.append(latent.clone())
    return trajectory

def ground_truth_trajectory(latent, num_steps=50):
    """
    Generate synthetic ground-truth trajectory (decay behavior).
    
    Args:
        latent (torch.Tensor): Initial latent vector
        num_steps (int): Number of steps
        
    Returns:
        List[torch.Tensor]: Ground truth trajectory
    """
    trajectory = [latent.clone()]
    for step in range(num_steps):
        latent = latent * 0.95  # ground-truth decaying behavior
        trajectory.append(latent.clone())
    return trajectory
