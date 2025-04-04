"""
Training module for APESGP experiments.

This module contains the implementation of the training procedures for the APESGP experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import os

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.apesgp_config import D, NUM_SUBFUNCTIONS, SUB_DIM
from preprocess import evaluate_full, evaluate_partial, generate_candidate

def ensure_directory(directory):
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)

def run_experiment_1(num_iterations=50, use_baseline=False):
    """
    Run Experiment 1.
    If use_baseline is True, always perform full evaluations.
    Otherwise, alternate between full and partial evaluations using a simple rule.
    A GP surrogate is updated after each evaluation.
    A progress curve (best objective value vs. cumulative cost) is plotted and saved as a PDF.
    """
    print("\n[Experiment 1] Starting {} evaluation run...".format(
        "baseline (full evaluations only)" if use_baseline else "APESGP (mixed evaluations)"))
    
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(
        length_scale=np.ones(D), 
        length_scale_bounds=(0.1, 10.0)
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True)
    
    objective_values = []
    cumulative_cost = []
    X_observed = []
    y_observed = []
    total_cost = 0.0

    for iter in range(num_iterations):
        x_candidate = generate_candidate()
        
        if use_baseline or (iter % 5 == 0):  
            y = evaluate_full(x_candidate)
            total_cost += 1.0
            eval_mode = "Full"
        else:
            idx = np.random.randint(NUM_SUBFUNCTIONS)
            y = evaluate_partial(x_candidate, idx)
            total_cost += 0.2
            eval_mode = "Partial"
        
        X_observed.append(x_candidate)
        y_observed.append(y)
        gp.fit(np.array(X_observed), np.array(y_observed))
        
        best_y = np.min(y_observed)
        objective_values.append(best_y)
        cumulative_cost.append(total_cost)
        
        print("Iteration {}: Mode = {:7s}, y = {:.4f}, best_y = {:.4f}, cumulative cost = {:.2f}".format(
            iter, eval_mode, y, best_y, total_cost))
    
    ensure_directory("logs")
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_cost, objective_values, marker='o', label="{}".format("Baseline" if use_baseline else "APESGP"))
    plt.xlabel("Cumulative Cost")
    plt.ylabel("Best Objective Value")
    plt.title("Experiment 1: Convergence Curve")
    plt.legend()
    fname = "logs/training_loss_{}_pair1.pdf".format("baseline" if use_baseline else "apesgp")
    plt.savefig(fname, format='pdf', dpi=300, bbox_inches='tight')
    print("Experiment 1 plot saved as '{}'.".format(fname))
    plt.close()
    
    return objective_values, cumulative_cost

def create_gp_variant(variant, dimension):
    """
    Create a GP surrogate with variant-specific lengthscale settings.
    Variant A: lengthscale ~ 1/D, Variant B: ~ 1/sqrt(D), Variant C: fixed value (control).
    """
    if variant == 'A':
        init_lengthscale = np.ones(dimension) * (1.0 / dimension)
        bounds = (np.ones(dimension) * (1.0 / (2*dimension)), np.ones(dimension) * (2.0 / dimension))
    elif variant == 'B':
        init_lengthscale = np.ones(dimension) * (1.0 / np.sqrt(dimension))
        bounds = (np.ones(dimension) * (1.0 / (2*np.sqrt(dimension))), np.ones(dimension) * (2.0 / np.sqrt(dimension)))
    else:  # Variant 'C': Fixed value as control.
        init_lengthscale = np.ones(dimension) * 1.0
        bounds = (np.ones(dimension) * 0.1, np.ones(dimension) * 10.0)
    
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(
        length_scale=init_lengthscale, 
        length_scale_bounds=(0.1, 10.0)
    )
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True)
    return gp_model
