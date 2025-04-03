"""
Evaluation functions for MML-BO experiments.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.train import SurrogateModel, optimize_baseline, optimize_mml_bo

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

os.makedirs('logs', exist_ok=True)

def multi_step_lookahead(model, x, levels=3):
    """
    Multi-step lookahead function using the surrogate model.
    
    Args:
        model (SurrogateModel): Surrogate model
        x (torch.Tensor): Current point
        levels (int): Number of levels for multi-level estimation
        
    Returns:
        list: Uncertainties at each level
    """
    uncertainties = []
    x_candidate = x.clone()
    
    for level in range(1, levels+1):
        x_candidate = x_candidate + (0.1/level) * torch.randn_like(x_candidate)
        _, unc = model(x_candidate)
        uncertainties.append(unc.item())
    
    return uncertainties

def experiment1(funcs, config):
    """
    Experiment 1: Performance on Synthetic Multi-Task Optimization with Heterogeneous Noise.
    
    Args:
        funcs (dict): Dictionary of synthetic functions
        config (dict): Configuration parameters
        
    Returns:
        dict: Results of the experiment
    """
    print("========== Experiment 1: Synthetic Multi-Task Optimization ==========")
    
    iters = config['iters']
    init_val = np.array(config['init_val'])
    
    results = {}
    
    quadratic_func = lambda x: funcs["quadratic"](x, config['noise_std_quad'])
    history_baseline_quad = optimize_baseline(quadratic_func, init_val, iters)
    history_mml_bo_quad = optimize_mml_bo(quadratic_func, init_val, iters, config['levels'])
    
    plt.figure()
    plt.plot(history_baseline_quad, label="MALIBO (baseline)")
    plt.plot(history_mml_bo_quad, label="MML-BO (multi-step)")
    plt.xlabel("Iteration")
    plt.ylabel("Best function value")
    plt.title("Optimization on Quadratic Function with Noise")
    plt.legend()
    fname = "logs/training_loss_quadratic.pdf"
    plt.savefig(fname)
    print(f"[Experiment 1] Saved plot as: {fname}")
    plt.close()
    
    sinusoidal_func = lambda x: funcs["sinusoidal"](x, config['noise_std_sin'])
    history_baseline_sin = optimize_baseline(sinusoidal_func, init_val, iters)
    history_mml_bo_sin = optimize_mml_bo(sinusoidal_func, init_val, iters, config['levels'])
    
    plt.figure()
    plt.plot(history_baseline_sin, label="MALIBO (baseline)")
    plt.plot(history_mml_bo_sin, label="MML-BO (multi-step)")
    plt.xlabel("Iteration")
    plt.ylabel("Best function value")
    plt.title("Optimization on Sinusoidal Function with Noise")
    plt.legend()
    fname = "logs/training_loss_sinusoidal.pdf"
    plt.savefig(fname)
    print(f"[Experiment 1] Saved plot as: {fname}")
    plt.close()
    
    results['quadratic'] = {
        'baseline': history_baseline_quad,
        'mml_bo': history_mml_bo_quad
    }
    
    results['sinusoidal'] = {
        'baseline': history_baseline_sin,
        'mml_bo': history_mml_bo_sin
    }
    
    return results

def experiment2(config):
    """
    Experiment 2: Adaptive Balance of Exploration and Exploitation via Multi-Step Lookahead.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        dict: Results of the experiment
    """
    print("========== Experiment 2: Adaptive Exploration-Exploitation ==========")
    
    model = SurrogateModel()
    x = torch.tensor([[0.0]])
    fixed_tau = config['fixed_tau']
    iters = config['iters']
    
    history_fixed = []
    history_adaptive = []
    tau_alloc = []  # record adaptive allocations
    
    for i in range(iters):
        _, unc_fixed = model(x)
        history_fixed.append(unc_fixed.item() * fixed_tau)
        
        uncertainties = multi_step_lookahead(model, x, levels=config['levels'])
        tau_adaptive = np.mean(uncertainties)
        tau_alloc.append(tau_adaptive)
        history_adaptive.append(tau_adaptive)
        
        if i % max(1, iters//10) == 0:
            print(f"[Experiment 2] Iteration {i}: fixed_tau_metric = {unc_fixed.item()*fixed_tau:.4f}, adaptive_tau = {tau_adaptive:.4f}")
        
        x = x + 0.1 * torch.randn_like(x)
    
    plt.figure()
    plt.plot(history_fixed, label="Fixed Exploration (MALIBO)")
    plt.plot(history_adaptive, label="Adaptive Exploration (MML-BO)")
    plt.xlabel("Iteration")
    plt.ylabel("Exploration Parameter (tau)")
    plt.title("Adaptive vs. Fixed Exploration-Exploitation Trade-off")
    plt.legend()
    fname = "logs/accuracy_adaptive_vs_fixed.pdf"
    plt.savefig(fname)
    print(f"[Experiment 2] Saved plot as: {fname}")
    plt.close()
    
    plt.figure()
    plt.plot(tau_alloc, label="Adaptive MLMC Lookahead Allocations")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Predicted Uncertainty")
    plt.title("MLMC Uncertainty Estimates per Iteration")
    plt.legend()
    fname = "logs/accuracy_mlmc_uncertainty_alloc_pair1.pdf"
    plt.savefig(fname)
    print(f"[Experiment 2] Saved plot as: {fname}")
    plt.close()
    
    return {
        'fixed': history_fixed,
        'adaptive': history_adaptive,
        'tau_alloc': tau_alloc
    }

def experiment3(data, target, config):
    """
    Experiment 3: Quality of Uncertainty Quantification through Meta-Learned Task Embeddings.
    
    Args:
        data (torch.Tensor): Input data
        target (torch.Tensor): Target values
        config (dict): Configuration parameters
        
    Returns:
        dict: Results of the experiment
    """
    from src.train import train_task_encoder
    
    _, loss_unimodal = train_task_encoder(
        data, target, 
        use_richer_uncertainty=False, 
        epochs=config['epochs'],
        lr=config['lr']
    )
    
    _, loss_richer = train_task_encoder(
        data, target, 
        use_richer_uncertainty=True, 
        epochs=config['epochs'],
        lr=config['lr']
    )
    
    plt.figure()
    plt.plot(loss_unimodal, label="Unimodal Gaussian prior")
    plt.plot(loss_richer, label="Richer uncertainty (Gaussian mixture)")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Meta-Learned Task Embedding: Uncertainty Quantification")
    plt.legend()
    fname = "logs/training_loss_meta_uncertainty.pdf"
    plt.savefig(fname)
    print(f"[Experiment 3] Saved plot as: {fname}")
    plt.close()
    
    return {
        'unimodal_loss': loss_unimodal,
        'richer_loss': loss_richer
    }
