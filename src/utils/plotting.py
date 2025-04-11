"""
Utility functions for creating and saving plots for the NSRPP experiments.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def save_figure(filename, output_dir="logs/figures"):
    """
    Save figure as a high-quality PDF suitable for academic papers.
    
    Args:
        filename (str): Name of the file (without extension)
        output_dir (str): Directory to save the figure
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Figure saved as {filepath}")
    plt.close()

def plot_risk_convergence(nsrpp_risk_history, bandit_risk_history, output_dir="logs/figures"):
    """
    Plot risk convergence curves for NSRPP and baseline methods.
    
    Args:
        nsrpp_risk_history (list): Risk history for NSRPP method
        bandit_risk_history (list): Risk history for baseline bandit method
        output_dir (str): Directory to save the figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(nsrpp_risk_history, label="NSRPP")
    plt.plot(bandit_risk_history, label="Two-Level Bandit", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("True Risk")
    plt.title("Risk Convergence: NSRPP vs. Two-Level Bandit")
    plt.legend()
    save_figure("risk_convergence_baseline_pair1", output_dir)

def plot_surrogate_ablation(results, output_dir="logs/figures"):
    """
    Plot bar chart comparing validation MSE for different surrogate architectures.
    
    Args:
        results (dict): Dictionary mapping surrogate names to MSE values
        output_dir (str): Directory to save the figure
    """
    plt.figure(figsize=(6, 4))
    variant_names = list(results.keys())
    mses = [results[name] for name in variant_names]
    plt.bar(variant_names, mses, color=["skyblue", "lightgreen", "salmon"])
    plt.xlabel("Surrogate Variant")
    plt.ylabel("Validation MSE")
    plt.title("Ablation Study on Surrogate Architectures")
    save_figure("surrogate_ablation_validation_mse_pair1", output_dir)

def plot_ucb_comparison(ucb_data, pe_data, output_dir="logs/figures"):
    """
    Create a two-panel plot comparing UCB and point estimate methods.
    
    Args:
        ucb_data (tuple): (thetas, risks, uncertainties) for UCB method
        pe_data (tuple): (thetas, risks, uncertainties) for Point Estimate method
        output_dir (str): Directory to save the figure
    """
    iterations = range(len(ucb_data[1]))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, ucb_data[1], label="UCB")
    plt.plot(iterations, pe_data[1], label="PointEstimate", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("True Risk")
    plt.title("Risk Convergence Comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iterations, ucb_data[0], label="UCB")
    plt.plot(iterations, pe_data[0], label="PointEstimate", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Theta")
    plt.title("Theta Trajectory Comparison")
    plt.legend()

    plt.tight_layout()
    save_figure("ucb_vs_point_convergence_pair1", output_dir)

def plot_nelder_mead_comparison(nsrpp_risk_history, nm_risk_history, output_dir="logs/figures"):
    """
    Plot risk convergence curves comparing NSRPP and Nelder-Mead methods.
    
    Args:
        nsrpp_risk_history (list): Risk history for NSRPP method
        nm_risk_history (list): Risk history for Nelder-Mead method
        output_dir (str): Directory to save the figure
    """
    plt.figure(figsize=(8, 6))
    iterations = range(len(nsrpp_risk_history))
    plt.plot(iterations, nsrpp_risk_history, label="NSRPP", color="blue")
    plt.plot(iterations, nm_risk_history, label="Nelder-Mead", color="red", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("True Risk")
    plt.title("Risk Convergence: NSRPP vs. Nelder-Mead")
    plt.legend()
    save_figure("nsrpp_vs_nelder_mead_comparison", output_dir)
