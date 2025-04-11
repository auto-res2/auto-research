"""
Model training functions for NSRPP experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from src.utils.models import SurrogateNet, OneLayerSurrogate, MultiLayerSurrogate, PretrainedSurrogate
from src.utils.data_generation import estimate_risk

def train_surrogate_model(model, train_loader, num_epochs=20, lr=1e-3):
    """
    Train a given surrogate model on provided DataLoader.
    
    Args:
        model (nn.Module): Surrogate model to train
        train_loader (DataLoader): DataLoader containing training data
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
    
    Returns:
        nn.Module: Trained model
    """
    optimizer_model = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer_model.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer_model.step()
            
    return model

def ucb_acquisition(surrogate, theta, beta=0.2, mc_runs=20):
    """
    Use Monte Carlo dropout on the surrogate to obtain the mean and uncertainty (std)
    for the risk prediction.
    
    Args:
        surrogate (nn.Module): Trained surrogate model
        theta (float): Current theta value
        beta (float): UCB exploration parameter
        mc_runs (int): Number of Monte Carlo samples
    
    Returns:
        tuple: (acquisition_value, mean_pred, std_pred)
    """
    surrogate.train()  # Ensures dropout is active.
    preds = []
    input_tensor = torch.tensor([[theta]], dtype=torch.float32)
    for _ in range(mc_runs):
        preds.append(surrogate(input_tensor).item())
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    acquisition_value = mean_pred - beta * std_pred
    return acquisition_value, mean_pred, std_pred

def run_nsrpp(num_iters=30, delta=0.1, learning_rate=0.1, init_theta=0.0):
    """
    Runs NSRPP method: trains a surrogate and uses UCB-like update.
    
    Args:
        num_iters (int): Number of iterations
        delta (float): Delta parameter for comparison with baseline
        learning_rate (float): Learning rate for parameter updates
        init_theta (float): Initial theta value
    
    Returns:
        list: Risk history for the NSRPP method
    """
    theta_nsrpp = init_theta
    surrogate = SurrogateNet()
    optimizer_surrogate = optim.Adam(surrogate.parameters(), lr=1e-3)
    nsrpp_risk_history = []
    
    for it in range(num_iters):
        true_risk = estimate_risk(theta_nsrpp)
        nsrpp_risk_history.append(true_risk)
        print(f"NSRPP Iter {it}: theta = {theta_nsrpp:.4f}, true risk = {true_risk:.4f}")

        surrogate.train()
        input_tensor = torch.tensor([[theta_nsrpp]], dtype=torch.float32)
        label = torch.tensor([[true_risk]], dtype=torch.float32)
        for _ in range(10):
            optimizer_surrogate.zero_grad()
            pred = surrogate(input_tensor)
            loss = nn.MSELoss()(pred, label)
            loss.backward()
            optimizer_surrogate.step()

        acquisition_value, mean_pred, std_pred = ucb_acquisition(surrogate, theta_nsrpp)
        theta_nsrpp = theta_nsrpp - learning_rate * (mean_pred - std_pred)
        
    return nsrpp_risk_history

def run_baseline_bandit(num_iters=30, delta=0.1, learning_rate=0.1, init_theta=0.0):
    """
    Runs baseline two-level bandit method using two-point zero-order gradient estimation.
    
    Args:
        num_iters (int): Number of iterations
        delta (float): Delta parameter for gradient estimation
        learning_rate (float): Learning rate for parameter updates
        init_theta (float): Initial theta value
    
    Returns:
        list: Risk history for the baseline method
    """
    theta_bandit = init_theta
    bandit_risk_history = []
    
    for it in range(num_iters):
        risk_plus = estimate_risk(theta_bandit + delta)
        risk_minus = estimate_risk(theta_bandit - delta)
        grad_est = (risk_plus - risk_minus) / (2 * delta)
        theta_bandit = theta_bandit - learning_rate * grad_est
        true_risk = estimate_risk(theta_bandit)
        bandit_risk_history.append(true_risk)
        print(f"Bandit Iter {it}: theta = {theta_bandit:.4f}, true risk = {true_risk:.4f}")
        
    return bandit_risk_history

def run_experiment3_variant(num_iters=30, learning_rate=0.1, init_theta=0.0, use_ucb=True):
    """
    Run the NSRPP outer loop update using either UCB or point estimate methods.
    
    Args:
        num_iters (int): Number of iterations
        learning_rate (float): Learning rate for parameter updates
        init_theta (float): Initial theta value
        use_ucb (bool): Whether to use UCB acquisition or point estimate
    
    Returns:
        tuple: (history_theta, history_risk, history_uncertainty)
    """
    method = "UCB" if use_ucb else "PointEstimate"
    print(f"Running Acquisition {method} variant ...")
    theta = init_theta
    surrogate = SurrogateNet()
    optimizer_surrogate = optim.Adam(surrogate.parameters(), lr=1e-3)
    history_theta = []
    history_risk = []
    history_uncertainty = []

    for it in range(num_iters):
        true_risk = estimate_risk(theta)
        history_theta.append(theta)
        history_risk.append(true_risk)

        print(f"{method} Iter {it}: theta = {theta:.4f}, true risk = {true_risk:.4f}")
        
        surrogate.train()
        input_tensor = torch.tensor([[theta]], dtype=torch.float32)
        label = torch.tensor([[true_risk]], dtype=torch.float32)
        for _ in range(5):
            optimizer_surrogate.zero_grad()
            pred = surrogate(input_tensor)
            loss = nn.MSELoss()(pred, label)
            loss.backward()
            optimizer_surrogate.step()

        if use_ucb:
            acq_value, mean_pred, std_pred = ucb_acquisition(surrogate, theta, beta=0.2, mc_runs=20)
            history_uncertainty.append(std_pred)
            candidate_theta = theta - learning_rate * (mean_pred - 0.2 * std_pred)
        else:
            surrogate.eval()
            with torch.no_grad():
                mean_pred = surrogate(input_tensor).item()
            history_uncertainty.append(0.0)
            candidate_theta = theta - learning_rate * mean_pred

        theta = candidate_theta

    return history_theta, history_risk, history_uncertainty
