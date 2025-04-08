"""
Training module for the NTEC-G experiment.
Implements the Base Method and NTEC-G guidance approaches.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time

class DiffusionGuidanceModel(torch.nn.Module):
    """
    Dummy guidance model for diffusion process.
    In a real implementation, this would be a more complex neural network.
    """
    def __init__(self):
        super(DiffusionGuidanceModel, self).__init__()
        
    def guidance_update(self, state):
        """
        A toy update to simulate diffusion guidance.
        Uses a small update based on the hyperbolic tangent.
        
        Args:
            state: Current state tensor
            
        Returns:
            torch.Tensor: Updated state after guidance step
        """
        return state - 0.1 * torch.tanh(state)

def base_method_guidance(x, model, epsilon=1e-4, max_iter=50):
    """
    Standard iterative fixed-point guidance (Base Method).
    
    Args:
        x: Input state tensor
        model: Guidance model
        epsilon: Convergence threshold
        max_iter: Maximum iterations
        
    Returns:
        tuple: (final_state, iterations_taken, norm_differences)
    """
    iter_num = 0
    prev = x.clone()
    norm_diffs = []
    
    while iter_num < max_iter:
        x = model.guidance_update(x)  # standard update step
        norm_diff = torch.norm(x - prev).item()
        norm_diffs.append(norm_diff)
        print(f"[Base Method] Iter: {iter_num}, norm diff: {norm_diff:.6f}")
        
        if norm_diff < epsilon:
            break
            
        prev = x.clone()
        iter_num += 1
        
    return x, iter_num, norm_diffs

def ntec_g_guidance(x, model, probe_steps=3, epsilon=1e-4):
    """
    NTK-based extrapolation guidance (NTEC-G).
    Uses a few probe iterations to predict the converged correction.
    
    Args:
        x: Input state tensor
        model: Guidance model
        probe_steps: Number of probe iterations
        epsilon: Convergence threshold
        
    Returns:
        tuple: (extrapolated_state, probe_norm_differences)
    """
    probes = []
    prev = x.clone()
    probe_norms = []
    
    for i in range(probe_steps):
        x = model.guidance_update(x)
        probes.append(x.clone())
        norm_diff = torch.norm(x - prev).item()
        probe_norms.append(norm_diff)
        print(f"[NTEC-G] Probe Iter: {i}, norm diff: {norm_diff:.6f}")
        prev = x.clone()
    
    if len(probes) >= 2:
        extrapolated = probes[-1] + (probes[-1] - probes[-2])
        print("[NTEC-G] Extrapolation performed using last two probe iterations.")
    else:
        extrapolated = probes[-1]
        
    return extrapolated, probe_norms

def geometric_regularization(z, r_target=1.0, lambda_reg=0.1):
    """
    Compute a radial regularization term for hyperbolic-inspired geometric constraints.
    
    Args:
        z: Latent representations
        r_target: Target radius
        lambda_reg: Regularization strength
        
    Returns:
        torch.Tensor: Regularization loss
    """
    norms = torch.norm(z, p=2, dim=1)
    radial_loss = F.mse_loss(norms, torch.full_like(norms, r_target))
    return lambda_reg * radial_loss

def diffusion_step_with_regularization(state, model, lambda_reg=0.1, use_reg=False):
    """
    One diffusion guidance step with optional geometric regularization.
    
    Args:
        state: Current state tensor
        model: Guidance model
        lambda_reg: Regularization strength
        use_reg: Whether to use regularization
        
    Returns:
        torch.Tensor: Updated state after guidance and regularization
    """
    state_updated = model.guidance_update(state)
    
    if use_reg:
        reg_loss = geometric_regularization(state_updated, lambda_reg=lambda_reg)
        
        reg_effect = 0.01 * reg_loss
        reg_effect = torch.clamp(reg_effect, max=1.0)  # Prevent extreme updates
        
        state_updated = state_updated - reg_effect
        print(f"[Diffusion with Reg] Regularization loss: {reg_loss.item():.6f}")
        
    return state_updated
