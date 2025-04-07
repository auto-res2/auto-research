"""
Utility functions for diffusion-based purification processes.
"""
import torch

def purify_fixed(x, lambda_value=0.5):
    """
    Fixed mixing Purify++ process with a fixed lambda value.
    
    Args:
        x (torch.Tensor): Input tensor to purify
        lambda_value (float): Fixed mixing coefficient
        
    Returns:
        torch.Tensor: Purified tensor
    """
    noise = torch.randn_like(x) * lambda_value
    return x + noise

def cov_purify_dynamic(x):
    """
    Cov-Purify++: dynamic mixing based on local covariance estimation.
    
    Args:
        x (torch.Tensor): Input tensor to purify
        
    Returns:
        torch.Tensor: Purified tensor
    """
    local_variance = torch.var(x, dim=[2, 3], keepdim=True)
    dynamic_lambda = torch.sigmoid(local_variance)  # dynamic mixing strength
    noise = torch.randn_like(x) * dynamic_lambda
    return x + noise

def adaptive_reverse_diffusion(x, num_steps=50, tol=1e-3):
    """
    Adaptive reverse diffusion process using covariance-based step size adaptation.
    
    Args:
        x (torch.Tensor): Input tensor to purify
        num_steps (int): Maximum number of diffusion steps
        tol (float): Tolerance for early stopping
        
    Returns:
        tuple: (purified tensor, list of step sizes)
    """
    purified = x.clone()
    step = 0
    current_time = 1.0  # simulated diffusion time (from t=1 to t=0)
    step_sizes = []  # log step size chosen at each iteration
    
    while current_time > 0 and step < num_steps:
        cov_estimate = torch.var(purified, dim=[2, 3], keepdim=True)
        base = 0.05
        step_size = base * torch.exp(-cov_estimate.mean()).item()
        step_sizes.append(step_size)
        
        noise_correction = -purified * step_size  # correction term
        purified = purified + noise_correction
        
        current_time -= step_size
        step += 1
    
    return purified, step_sizes

def fixed_reverse_diffusion(x, num_steps=50):
    """
    Fixed reverse diffusion process using a constant step size.
    
    Args:
        x (torch.Tensor): Input tensor to purify
        num_steps (int): Number of diffusion steps
        
    Returns:
        torch.Tensor: Purified tensor
    """
    purified = x.clone()
    step_size = 1.0 / num_steps
    for _ in range(num_steps):
        noise_correction = -purified * step_size
        purified = purified + noise_correction
    return purified
