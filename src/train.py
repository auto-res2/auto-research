#!/usr/bin/env python3
"""
Implementation of models and training functions for HACP experiments.
This module includes the model definitions for:
- BaseMethod: Standard contrastive representation learning
- HACP: Hierarchical Adaptive Contrastive Planning
- HACP_Ablated: HACP with planning module removed for ablation study
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from collections import deque


# Model Definitions
class PerceptualBackbone(nn.Module):
    """Perceptual backbone shared by all methods."""
    def __init__(self, input_shape, hidden_size):
        super(PerceptualBackbone, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), hidden_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)


class BaseMethod(nn.Module):
    """Base method: only standard contrastive representation."""
    def __init__(self, input_shape, hidden_size, output_size):
        super(BaseMethod, self).__init__()
        self.backbone = PerceptualBackbone(input_shape, hidden_size)
        self.policy = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.policy(features)
        return logits, features


class HopfieldModule(nn.Module):
    """Planning Module: Simplified version of modern Hopfield network for adaptive clustering."""
    def __init__(self, feature_size, temperature=1.0):
        super(HopfieldModule, self).__init__()
        self.temperature = temperature
        # Parameterized weight to simulate adaptive clustering dynamics
        self.weight = nn.Parameter(torch.randn(feature_size, feature_size))
    
    def forward(self, features):
        # Compute similarity matrix and apply temperature scaling
        sim = torch.matmul(features, self.weight)  # shape: [batch, feature_size]
        sim = sim / self.temperature
        attn = torch.softmax(sim, dim=-1)
        # Transpose features for correct matrix multiplication
        features_t = features.transpose(0, 1)  # [feature_size, batch]
        # Produce an abstract representation
        abstract_repr = torch.matmul(attn, features_t).transpose(0, 1)
        return abstract_repr


class HACP(nn.Module):
    """HACP extends BaseMethod with planning module and two-phase cycle."""
    def __init__(self, input_shape, hidden_size, output_size, planning_temperature=1.0):
        super(HACP, self).__init__()
        self.backbone = PerceptualBackbone(input_shape, hidden_size)
        self.policy = nn.Linear(hidden_size, output_size)
        self.hopfield = HopfieldModule(hidden_size, temperature=planning_temperature)
    
    def forward(self, x):
        features = self.backbone(x)
        # Phase 1: "Reason for Future" using adaptive clustering
        adaptive_features = self.hopfield(features)
        # Phase 2: "Act for Now" by combining standard and adaptive features
        combined_features = features + adaptive_features
        logits = self.policy(combined_features)
        return logits, combined_features


class AblatedHopfieldModule(nn.Module):
    """Ablated Planning Module that outputs random noise."""
    def __init__(self, feature_size):
        super(AblatedHopfieldModule, self).__init__()
        self.feature_size = feature_size
    
    def forward(self, features):
        # Return random tensor with same shape as features
        noise = torch.randn_like(features)
        return noise


class HACP_Ablated(nn.Module):
    """HACP model with planning module ablated."""
    def __init__(self, input_shape, hidden_size, output_size):
        super(HACP_Ablated, self).__init__()
        self.backbone = PerceptualBackbone(input_shape, hidden_size)
        self.policy = nn.Linear(hidden_size, output_size)
        self.hopfield = AblatedHopfieldModule(hidden_size)
    
    def forward(self, x):
        features = self.backbone(x)
        # Ablated planning: add random noise instead of adaptive features
        ablated_features = self.hopfield(features)
        combined_features = features + ablated_features
        logits = self.policy(combined_features)
        return logits, combined_features


# Training function
def train_model(model, env, num_episodes=100, lr=1e-3, gamma=0.99, verbose=True, device="cuda"):
    """
    Training loop for one RL model.
    Uses a dummy loss (negative reward) to simulate training.
    
    Args:
        model: Model to train
        env: Gymnasium environment
        num_episodes: Number of episodes to train for
        lr: Learning rate
        gamma: Discount factor
        verbose: Print progress
        device: Device to train on ("cuda" or "cpu")
        
    Returns:
        list: History of episodic rewards
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reward_history = []
    
    # Use fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Progress bar for training
    pbar = tqdm(range(num_episodes), desc="Training", disable=not verbose)
    
    for episode in pbar:
        obs, _ = env.reset()
        episodic_reward = 0
        done = False
        step = 0
        
        while not done:
            # Convert observation to tensor with batch dimension
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(obs_tensor)
            action_prob = torch.softmax(logits, dim=-1)
            # Sample action according to computed probabilities
            action = torch.multinomial(action_prob, num_samples=1).item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episodic_reward += reward
            step += 1

            # In absence of full RL algorithm, use negative reward as dummy loss
            loss = -torch.tensor(reward, dtype=torch.float32, requires_grad=True).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            obs = next_obs  # Move to next state
        
        reward_history.append(episodic_reward)
        pbar.set_postfix({"reward": episodic_reward})
        
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_name = model.__class__.__name__
    torch.save(model.state_dict(), f"models/{model_name}_trained.pt")
    
    return reward_history
