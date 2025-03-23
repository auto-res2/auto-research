"""
Model evaluation for ABS-Diff experiments.

This script implements the evaluation metrics and 
testing routines for the ABS-Diff model.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ks_2samp
from tqdm import tqdm


def test_dynamic_noise_conditioning(regime_classifier, adaptive_model, fixed_model):
    """
    Test dynamic noise conditioning vs. fixed noise scheduling.
    
    Args:
        regime_classifier: Regime classifier model
        adaptive_model: Model with dynamic noise conditioning
        fixed_model: Model with fixed noise scheduling
        
    Returns:
        loss_dynamic: Loss for dynamic conditioning
        loss_fixed: Loss for fixed scheduling
    """
    # Create dummy batch
    x = torch.randn(8, 3, 32, 32)
    t = torch.randint(0, 1000, (8,), device=x.device).float()
    
    loss_fn = nn.MSELoss()
    
    # Test dynamic noise conditioning
    noise_schedule_dynamic = torch.ones(8, device=x.device)
    noise = torch.randn_like(x)
    x_noised = x + noise * (t.view(-1, 1, 1, 1) / 1000.0)
    output_dynamic, updated_noise = adaptive_model(x_noised, t, noise_schedule_dynamic)
    loss_dynamic = loss_fn(output_dynamic, noise)
    print(f"Dynamic conditioning loss: {loss_dynamic.item():.6f}, Updated noise (mean): {updated_noise.mean().item():.4f}")
    
    # Test fixed noise scheduling
    from src.train import fixed_noise_schedule
    noise_schedule_fixed = torch.tensor([fixed_noise_schedule(ti.item()) for ti in t], device=x.device)
    output_fixed, fixed_noise_val = fixed_model(x_noised, t, noise_schedule_fixed)
    loss_fixed = loss_fn(output_fixed, noise)
    print(f"Fixed scheduling loss: {loss_fixed.item():.6f}, Fixed noise schedule (mean): {noise_schedule_fixed.mean().item():.4f}")
    
    return loss_dynamic, loss_fixed


def run_reverse_sde(initial_state, regime_tag, num_steps=10):
    """
    Run reverse SDE simulation for different regimes.
    
    Args:
        initial_state: Initial state
        regime_tag: Regime type ("memorization" or "generalization")
        num_steps: Number of simulation steps
        
    Returns:
        states: List of states during simulation
        noise_scales: List of noise scales used
    """
    from src.train import AdaptiveSDESolver
    
    print(f"\nRunning reverse SDE simulation for regime: {regime_tag}")
    solver = AdaptiveSDESolver(regime=regime_tag)
    states = [initial_state.detach().cpu().numpy()]
    state = initial_state
    noise_scales = []
    
    for t in range(num_steps):
        state, noise_scale = solver(state, t)
        noise_scales.append(noise_scale)
        states.append(state.detach().cpu().numpy())
        if t == 0:
            print(f"Step {t}: noise_scale = {noise_scale}, sample state mean = {state.mean().item():.4f}")
    
    print(f"Completed reverse SDE for {regime_tag}.")
    return states, noise_scales


def compare_sde_regimes():
    """
    Compare SDE solvers across different regimes.
    
    Returns:
        statistic: KS statistic for comparison
        p_value: P-value for comparison
    """
    # Simulate an initial state
    initial_state = torch.randn(16, 3, 32, 32)
    
    states_mem, noise_scales_mem = run_reverse_sde(initial_state.clone(), regime_tag="memorization", num_steps=50)
    states_gen, noise_scales_gen = run_reverse_sde(initial_state.clone(), regime_tag="generalization", num_steps=50)
    
    # Compare final state distributions using Kolmogorov–Smirnov test
    final_state_mem = states_mem[-1].flatten()
    final_state_gen = states_gen[-1].flatten()
    statistic, p_value = ks_2samp(final_state_mem, final_state_gen)
    
    print("\nKS test between memorization and generalization final states:")
    print(f"KS Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
    
    # Log average noise scales
    print(f"Average noise_scale in memorization regime: {np.mean(noise_scales_mem):.4f}")
    print(f"Average noise_scale in generalization regime: {np.mean(noise_scales_gen):.4f}")
    
    return statistic, p_value
