#!/usr/bin/env python3
"""
Evaluation code for HACP experiments.
This module contains functions for:
- Collecting intermediate representations for visualization
- Visualizing high-dimensional representations with t-SNE
- Running ablation studies
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def collect_representations(model, env, num_steps=200, device="cuda"):
    """
    Collect intermediate representations for visualization.
    
    Args:
        model: Trained model
        env: Gymnasium environment
        num_steps: Number of steps to collect representations
        device: Device to run model on ("cuda" or "cpu")
    
    Returns:
        tuple: (representations, labels) where representations is a numpy array
               of feature vectors and labels is a numpy array of binary labels.
    """
    model = model.to(device)
    representations = []
    labels = []
    obs, _ = env.reset()
    step = 0
    done = False
    
    while step < num_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        # Get intermediate representation from model (second output)
        _, features = model(obs_tensor)
        # Detach and store features as numpy array
        representations.append(features.detach().cpu().numpy().squeeze())
        
        # For controlled labeling, create dummy label
        # Label = 1 if sum(observation) exceeds median value, else 0
        obs_sum = np.sum(obs)
        dummy_median = np.median(obs) if np.prod(obs.shape) > 0 else 0.5
        label = 1 if obs_sum > dummy_median else 0
        labels.append(label)
        
        # Generate action to move in environment
        logits, _ = model(obs_tensor)
        action_prob = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_prob, num_samples=1).item()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
        step += 1
    
    return np.array(representations), np.array(labels)

def plot_tsne(repr_base, labels_base, repr_hacp, labels_hacp):
    """
    Reduce high-dimensional representations to 2D using t-SNE and plot them.
    
    Args:
        repr_base: Representations from Base Method
        labels_base: Labels for Base Method representations
        repr_hacp: Representations from HACP
        labels_hacp: Labels for HACP representations
    """
    os.makedirs("logs", exist_ok=True)
    
    tsne = TSNE(n_components=2, random_state=42)
    base_2d = tsne.fit_transform(repr_base)
    hacp_2d = tsne.fit_transform(repr_hacp)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=base_2d[:, 0], y=base_2d[:, 1], hue=labels_base, palette="viridis", ax=ax[0])
    ax[0].set_title("Base Method Representations (t-SNE)")
    sns.scatterplot(x=hacp_2d[:, 0], y=hacp_2d[:, 1], hue=labels_hacp, palette="viridis", ax=ax[1])
    ax[1].set_title("HACP Representations (t-SNE)")
    
    # Save plot to logs directory
    plt.tight_layout()
    plt.savefig("logs/tsne_visualization.png")
    
    return fig

def evaluate_model(model, env, num_episodes=10, device="cuda"):
    """
    Evaluate a trained model by running it in the environment and recording rewards.
    
    Args:
        model: Trained model
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
        device: Device to run model on ("cuda" or "cpu")
    
    Returns:
        float: Average reward across episodes
    """
    model = model.to(device)
    rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(obs_tensor)
            action_prob = torch.softmax(logits, dim=-1)
            action = torch.multinomial(action_prob, num_samples=1).item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)
