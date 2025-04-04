"""
Evaluation module for APESGP experiments.

This module contains the implementation of the evaluation procedures for the APESGP experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.apesgp_config import D, NUM_SUBFUNCTIONS, COST_PARTIAL_DEFAULT, COST_FULL_DEFAULT
from preprocess import evaluate_full, evaluate_partial, generate_candidate
from train import create_gp_variant, ensure_directory

def run_experiment_2(num_iterations=50):
    """
    Run Experiment 2.
    Vary the cost ratios (c1, c2) and simulate an acquisition function that selects either a partial or full evaluation
    based on a cost-aware criterion (expected improvement per unit cost).
    Log and plot the convergence curve and print the number of evaluation types chosen.
    """
    print("\n[Experiment 2] Starting cost-aware acquisition function analysis...")
    cost_ratios = [(0.2, 1.0), (0.3, 1.0), (0.1, 1.0)]
    results = {}
    
    plt.figure(figsize=(10, 6))
    for c1, c2 in cost_ratios:
        partial_count = 0
        full_count = 0
        total_cost = 0.0
        obj_values = []
        cost_history = []
        X_obs = []
        y_obs = []

        for i in range(num_iterations):
            x_cand = generate_candidate()
            ei_full = np.random.rand()
            ei_partial = np.random.rand()

            score_full = ei_full / c2
            score_partial = ei_partial / c1

            if score_partial > score_full:
                idx = np.random.randint(NUM_SUBFUNCTIONS)
                y = evaluate_partial(x_cand, idx, cost_partial=c1)
                total_cost += c1
                partial_count += 1
                eval_type = "Partial"
            else:
                y = evaluate_full(x_cand, cost_full=c2)
                total_cost += c2
                full_count += 1
                eval_type = "Full"
            
            X_obs.append(x_cand)
            y_obs.append(y)
            current_best = np.min(y_obs)
            obj_values.append(current_best)
            cost_history.append(total_cost)
            
            print("Cost Ratio (c1, c2)=({}, {}), Iter {}: {} eval, y={:.4f}, best_y={:.4f}, cost={:.2f}".format(
                c1, c2, i, eval_type, y, current_best, total_cost))
        
        results[(c1, c2)] = {
            'partial_count': partial_count,
            'full_count': full_count,
            'obj_values': obj_values,
            'cost_history': cost_history,
        }
        plt.plot(cost_history, obj_values, marker='o', label="c1={}, c2={}".format(c1, c2))
    
    plt.xlabel("Cumulative Cost")
    plt.ylabel("Best Objective Value")
    plt.title("Experiment 2: Cost-Aware Acquisition")
    plt.legend()
    
    ensure_directory("logs")
    
    fname = "logs/accuracy_cost_awareness_pair1.pdf"
    plt.savefig(fname, format='pdf', dpi=300, bbox_inches='tight')
    print("Experiment 2 plot saved as '{}'.".format(fname))
    plt.close()
    
    for key, val in results.items():
        print("Cost ratio {}: partial evals = {}, full evals = {}".format(
            key, val['partial_count'], val['full_count']))
    
    return results

def run_experiment_3(num_iterations=50, num_seed_runs=3):
    """
    Run Experiment 3.
    Compare three GP variants (A, B, C) having different scaling in the lengthscale.
    For each variant (and over several seeds), track the best objective value per iteration.
    """
    print("\n[Experiment 3] Starting sensitivity analysis on GP hyperparameters...")
    variants = ['A', 'B', 'C']
    performance_results = {variant: [] for variant in variants}
    
    for seed in range(num_seed_runs):
        print(" Seed run {}...".format(seed))
        np.random.seed(seed)
        for variant in variants:
            gp_variant = create_gp_variant(variant, D)
            X_obs = []
            y_obs = []
            total_cost = 0.0
            obj_hist = []
            for i in range(num_iterations):
                x_cand = generate_candidate()
                if i % 5 == 0:
                    y = evaluate_full(x_cand)
                    total_cost += COST_FULL_DEFAULT
                    eval_type = "Full"
                else:
                    idx = np.random.randint(NUM_SUBFUNCTIONS)
                    y = evaluate_partial(x_cand, idx)
                    total_cost += COST_PARTIAL_DEFAULT
                    eval_type = "Partial"
                
                X_obs.append(x_cand)
                y_obs.append(y)
                gp_variant.fit(np.array(X_obs), np.array(y_obs))
                current_best = np.min(y_obs)
                obj_hist.append(current_best)
                
                print(" Variant {} | Iter {}: {} eval, y={:.4f}, best_y={:.4f}, cost={:.2f}".format(
                    variant, i, eval_type, y, current_best, total_cost))
            performance_results[variant].append(obj_hist)
    
    plt.figure(figsize=(10, 6))
    for variant in variants:
        performance_mean = np.mean(performance_results[variant], axis=0)
        plt.plot(range(num_iterations), performance_mean, marker='o', label="Variant {}".format(variant))
    
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value")
    plt.title("Experiment 3: GP Variant Convergence")
    plt.legend()
    
    ensure_directory("logs")
    
    fname = "logs/training_loss_dimscaled_pair1.pdf"
    plt.savefig(fname, format='pdf', dpi=300, bbox_inches='tight')
    print("Experiment 3 plot saved as '{}'.".format(fname))
    plt.close()
    
    return performance_results
