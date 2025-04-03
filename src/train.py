"""
Training models for MML-BO experiments.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SurrogateModel(nn.Module):
    """
    A dummy surrogate model that produces a prediction and uncertainty.
    """
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.fc = nn.Linear(1, 1)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (prediction, uncertainty) tensors
        """
        pred = self.fc(x)
        uncertainty = torch.abs(pred) * 0.1 + 0.05
        return pred, uncertainty

class TaskEncoder(nn.Module):
    """
    Meta-learning encoder network for task embeddings.
    """
    def __init__(self, output_dim=1, use_richer_uncertainty=False):
        super(TaskEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 20), 
            nn.ReLU(), 
            nn.Linear(20, output_dim)
        )
        self.use_richer_uncertainty = use_richer_uncertainty
        
        if self.use_richer_uncertainty:
            self.logits = nn.Linear(10, 2)
            self.means = nn.Linear(10, 2)
            self.log_scales = nn.Linear(10, 2)
            
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (embedding, uncertainty_model) where uncertainty_model is a distribution
        """
        embedding = self.fc(x)
        
        if self.use_richer_uncertainty:
            logits = self.logits(x)
            weights = torch.softmax(logits, dim=-1)  # mixture component weights
            means = self.means(x)
            log_scales = self.log_scales(x)
            scales = torch.exp(log_scales)
            
            mixture = dist.Categorical(weights)
            comp = dist.Normal(means, scales)
            gmm = dist.MixtureSameFamily(mixture, dist.Independent(comp, 0))
            return embedding, gmm
        else:
            gaussian = dist.Normal(embedding, 0.1)
            return embedding, gaussian

def optimize_baseline(func, init, iters=50, step_size=0.1):
    """
    Baseline optimization (MALIBO) simulated by a basic exploration update.
    
    Args:
        func (callable): Function to optimize
        init (np.ndarray): Initial point
        iters (int): Number of iterations
        step_size (float): Step size for random exploration
        
    Returns:
        list: History of best values found during optimization
    """
    x = init.copy()
    best_val = np.inf
    history = []
    
    print("[Experiment 1][Baseline] Starting optimization...")
    for i in range(iters):
        val = func(x)
        best_val = min(best_val, val)
        history.append(best_val)
        
        if i % max(1, iters//10) == 0:
            print(f"[Experiment 1][Baseline] Iteration {i}: best_val = {float(best_val):.4f}")
            
        x = x + step_size * np.random.randn(*x.shape)
    
    print("[Experiment 1][Baseline] Optimization finished!")
    return history

def optimize_mml_bo(func, init, iters=50, levels=3, step_size=0.1):
    """
    MML-BO optimization simulated with a multi-step lookahead.
    
    Args:
        func (callable): Function to optimize
        init (np.ndarray): Initial point
        iters (int): Number of iterations
        levels (int): Number of levels for multi-level estimation
        step_size (float): Step size for random exploration
        
    Returns:
        list: History of best values found during optimization
    """
    x = init.copy()
    best_val = np.inf
    history = []
    
    print("[Experiment 1][MML-BO] Starting multi-step optimization...")
    for i in range(iters):
        candidate_vals = []
        
        for level in range(1, levels+1):
            candidate = x + (step_size/level) * np.random.randn(*x.shape)
            candidate_val = func(candidate)
            candidate_vals.append(candidate_val)
            
        best_candidate_val = min(candidate_vals)
        best_val = min(best_val, best_candidate_val)
        history.append(best_val)
        
        if i % max(1, iters//10) == 0:
            print(f"[Experiment 1][MML-BO] Iteration {i}: best_candidate_val = {float(best_candidate_val):.4f}")
            
        x = x + step_size * np.random.randn(*x.shape)
    
    print("[Experiment 1][MML-BO] Optimization finished!")
    return history

def train_task_encoder(data, target, use_richer_uncertainty, epochs=100, lr=0.01, device=device):
    """
    Train the task encoder model.
    
    Args:
        data (torch.Tensor): Input data
        target (torch.Tensor): Target values
        use_richer_uncertainty (bool): Whether to use richer uncertainty (GMM)
        epochs (int): Number of training epochs
        lr (float): Learning rate
        device (torch.device): Device to run the training on
        
    Returns:
        tuple: (model, losses) where model is the trained encoder and losses is a list of loss values
    """
    data = data.to(device)
    target = target.to(device)
    
    encoder = TaskEncoder(use_richer_uncertainty=use_richer_uncertainty).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    losses = []
    
    print(f"[Experiment 3] Training TaskEncoder with use_richer_uncertainty={use_richer_uncertainty}")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()
        embedding, uncertainty_model = encoder(data)
        
        log_probs = uncertainty_model.log_prob(target.squeeze())
        loss = -log_probs.mean()
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % max(1, epochs//10) == 0:
            print(f"[Experiment 3] Epoch {epoch}: Loss = {loss.item():.4f}")
    
    os.makedirs('models', exist_ok=True)
    model_path = f'models/task_encoder_{"richer" if use_richer_uncertainty else "simple"}.pt'
    torch.save(encoder.state_dict(), model_path)
    
    return encoder, losses
