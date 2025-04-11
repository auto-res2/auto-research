"""
Model evaluation functions for NSRPP experiments.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from src.utils.models import OneLayerSurrogate, MultiLayerSurrogate, PretrainedSurrogate
from src.utils.plotting import plot_risk_convergence, plot_surrogate_ablation, plot_ucb_comparison
from src.train import train_surrogate_model

def evaluate_experiment1(nsrpp_risk_history, bandit_risk_history, output_dir="logs/figures"):
    """
    Evaluate results from Experiment 1 and generate plots.
    
    Args:
        nsrpp_risk_history (list): Risk history for NSRPP method
        bandit_risk_history (list): Risk history for baseline bandit method
        output_dir (str): Directory to save plots
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    plot_risk_convergence(nsrpp_risk_history, bandit_risk_history, output_dir)
    
    nsrpp_final_risk = nsrpp_risk_history[-1]
    bandit_final_risk = bandit_risk_history[-1]
    risk_reduction = (bandit_final_risk - nsrpp_final_risk) / bandit_final_risk * 100
    
    nsrpp_sample_usage = len(nsrpp_risk_history) * 500
    bandit_sample_usage = len(bandit_risk_history) * 1500  # 3x due to two-point sampling
    sample_efficiency = (bandit_sample_usage - nsrpp_sample_usage) / bandit_sample_usage * 100
    
    return {
        "nsrpp_final_risk": nsrpp_final_risk,
        "bandit_final_risk": bandit_final_risk,
        "risk_reduction_percent": risk_reduction,
        "sample_efficiency_percent": sample_efficiency
    }

def evaluate_experiment2(train_loader, val_dataset, output_dir="logs/figures"):
    """
    Evaluate results from Experiment 2 (surrogate ablation) and generate plots.
    
    Args:
        train_loader (DataLoader): DataLoader for training data
        val_dataset (Dataset): Dataset for validation
        output_dir (str): Directory to save plots
    
    Returns:
        dict: Dictionary mapping surrogate variant names to validation MSE
    """
    surrogate_variants = {
        "OneLayer": OneLayerSurrogate(),
        "MultiLayer": MultiLayerSurrogate(),
        "Pretrained": PretrainedSurrogate()
    }
    results = {}

    for name, model in surrogate_variants.items():
        print(f"Training surrogate variant: {name} ...")
        model = train_surrogate_model(model, train_loader, num_epochs=20, lr=1e-3)
        model.eval()
        with torch.no_grad():
            val_inputs = val_dataset.thetas
            val_targets = val_dataset.risks
            preds = model(val_inputs).numpy().flatten()
        mse = mean_squared_error(val_targets.numpy().flatten(), preds)
        results[name] = mse
        print(f"{name} surrogate validation MSE: {mse:.4f}")

    plot_surrogate_ablation(results, output_dir)
    return results

def evaluate_experiment3(ucb_history, pe_history, output_dir="logs/figures"):
    """
    Evaluate results from Experiment 3 (uncertainty-aware acquisition) and generate plots.
    
    Args:
        ucb_history (tuple): (thetas, risks, uncertainties) for UCB method
        pe_history (tuple): (thetas, risks, uncertainties) for Point Estimate method
        output_dir (str): Directory to save plots
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    plot_ucb_comparison(ucb_history, pe_history, output_dir)
    
    ucb_final_risk = ucb_history[1][-1]
    pe_final_risk = pe_history[1][-1]
    risk_difference = (pe_final_risk - ucb_final_risk) / pe_final_risk * 100
    
    return {
        "ucb_final_risk": ucb_final_risk,
        "pe_final_risk": pe_final_risk,
        "risk_reduction_percent": risk_difference
    }
