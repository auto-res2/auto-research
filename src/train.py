#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements the training components for the BI-SDICL (Bias-Integrated Sequential 
Decision In-Context Learner) experiments, including model architecture and training routines.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ================================
# Policy Networks
# ================================
class BI_SDICLPolicy(nn.Module):
    """
    A transformer-based policy that accepts demonstration tokens.
    When use_bias_conversion=True, a bias conversion module is applied to the demonstration embeddings.
    """
    def __init__(self, state_dim, action_dim, demo_embedding_dim=64, use_bias_conversion=True):
        super(BI_SDICLPolicy, self).__init__()
        self.use_bias_conversion = use_bias_conversion
        self.state_embedding = nn.Linear(state_dim, 128)
        # Transformer block: We use a simple MultiheadAttention layer with batch_first=True.
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # Feedforward decision layer: from attended features to actions.
        self.decision_head = nn.Linear(128, action_dim)
        # Module to convert demonstration embeddings, a simple MLP.
        self.bias_conversion = nn.Sequential(
            nn.Linear(demo_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, state, demo_tokens=None):
        """
        Forward pass of the BI-SDICL policy.
        
        Args:
            state: (batch, state_dim) - State tensor
            demo_tokens: (batch, num_demos, demo_embedding_dim) or None - Demonstration tokens
            
        Returns:
            Action logits
        """
        # Compute state embedding, add sequence dimension.
        state_emb = self.state_embedding(state).unsqueeze(1)  # (batch, 1, 128)
        if demo_tokens is not None:
            # Aggregate demonstration tokens (simple mean) over demonstrations.
            demo_agg = demo_tokens.mean(dim=1)  # (batch, demo_embedding_dim)
            if self.use_bias_conversion:
                bias = self.bias_conversion(demo_agg).unsqueeze(1)  # (batch, 1, 128)
            else:
                # For no conversion case, project to the same dimension as state embedding
                bias = nn.Linear(demo_agg.shape[-1], 128)(demo_agg).unsqueeze(1)  # (batch, 1, 128)
            # Concatenate bias token with state embedding.
            x = torch.cat([bias, state_emb], dim=1)
        else:
            x = state_emb

        # Apply self-attention using x as query, key, and value.
        attended, _ = self.attention(x, x, x)
        # Use the representation corresponding to actual state.
        out = attended[:, -1, :]  # (batch, 128)
        return self.decision_head(out)

    def get_bias_vector(self, state, demo_tokens):
        """
        Returns the computed bias vector from the bias conversion module.
        Used for diagnostic visualization.
        
        Args:
            state: (batch, state_dim) - State tensor
            demo_tokens: (batch, num_demos, demo_embedding_dim) - Demonstration tokens
            
        Returns:
            Bias vector
        """
        state_emb = self.state_embedding(state).unsqueeze(1)
        demo_agg = demo_tokens.mean(dim=1)
        bias = self.bias_conversion(demo_agg).unsqueeze(1)
        return bias

class BaseMethodPolicy(BI_SDICLPolicy):
    """
    For the Base Method we do not use demonstration tokens at all.
    """
    def forward(self, state, demo_tokens=None):
        """
        Forward pass of the Base Method policy.
        
        Args:
            state: (batch, state_dim) - State tensor
            demo_tokens: Ignored in this implementation
            
        Returns:
            Action logits
        """
        # Only use state information (do not concatenate any demonstration embedding)
        state_emb = self.state_embedding(state).unsqueeze(1)
        attended, _ = self.attention(state_emb, state_emb, state_emb)
        out = attended[:, -1, :]
        return self.decision_head(out)

# ================================
# Training Routines
# ================================
def train_agent(env, policy, optimizer, num_episodes=50, demo_tokens=None, print_interval=10):
    """
    Trains the given policy for num_episodes on the provided environment.
    Measures cumulative reward per episode.
    
    Args:
        env: Gym environment
        policy: Policy network to train
        optimizer: Optimizer for policy network
        num_episodes: Number of episodes to train for
        demo_tokens: Demonstration tokens for BI-SDICL
        print_interval: How often to print progress
        
    Returns:
        List of cumulative rewards per episode
    """
    cumulative_rewards = []
    policy.train()
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)  # shape: (1, state_dim)
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            logits = policy(state, demo_tokens)
            action = logits.argmax(dim=-1).item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Dummy loss: try to push the mean of logits close to reward (for demo purposes)
            loss = (logits.mean() - reward)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = torch.FloatTensor(next_state).unsqueeze(0)
            step_count += 1

        cumulative_rewards.append(episode_reward)
        if (episode+1) % print_interval == 0:
            print(f"[Episode {episode+1}/{num_episodes}] Reward: {episode_reward:.2f}")

    return cumulative_rewards

def train_with_bias_perturbation(env, policy, optimizer, demo_tokens, perturb=False, num_episodes=20):
    """
    Trains the policy with the option to perturb (e.g., zero out) the bias vector.
    Used for interpretability experiments.
    
    Args:
        env: Gym environment
        policy: BI-SDICL policy
        optimizer: Optimizer for policy
        demo_tokens: Demonstration tokens
        perturb: Whether to perturb the bias vector
        num_episodes: Number of episodes to train for
        
    Returns:
        List of cumulative rewards per episode
    """
    cumulative_rewards = []
    policy.train()
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        episode_reward = 0.0
        
        while not done:
            # Get bias vector from demonstration tokens
            bias = policy.get_bias_vector(state, demo_tokens)
            if perturb:
                # Zero out the bias vector if perturb is True
                bias = torch.zeros_like(bias)
                
            # Forward pass: combine bias and state embedding
            state_emb = policy.state_embedding(state).unsqueeze(1)
            x = torch.cat([bias, state_emb], dim=1)
            attended, _ = policy.attention(x, x, x)
            out = attended[:, -1, :]
            logits = policy.decision_head(out)
            
            action = logits.argmax(dim=-1).item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Compute loss and update policy
            loss = (logits.mean() - reward)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = torch.FloatTensor(next_state).unsqueeze(0)
            
        cumulative_rewards.append(episode_reward)
        if (episode+1) % 5 == 0:
            print(f"[Episode {episode+1}/{num_episodes}] Reward: {episode_reward:.2f}")
            
    return cumulative_rewards

def create_policy(state_dim, action_dim, policy_type="bi_sdicl", use_bias_conversion=True):
    """
    Creates a policy network of the specified type.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        policy_type: Type of policy to create ("bi_sdicl" or "base")
        use_bias_conversion: Whether to use bias conversion module (for BI-SDICL)
        
    Returns:
        Policy network
    """
    if policy_type == "bi_sdicl":
        return BI_SDICLPolicy(state_dim, action_dim, use_bias_conversion=use_bias_conversion)
    elif policy_type == "base":
        return BaseMethodPolicy(state_dim, action_dim)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

def save_model(policy, path):
    """
    Saves the policy network to the specified path.
    
    Args:
        policy: Policy network to save
        path: Path to save the model to
    """
    torch.save(policy.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(policy, path):
    """
    Loads the policy network from the specified path.
    
    Args:
        policy: Policy network to load into
        path: Path to load the model from
        
    Returns:
        Loaded policy network
    """
    policy.load_state_dict(torch.load(path))
    policy.eval()
    print(f"Model loaded from {path}")
    return policy

if __name__ == "__main__":
    # Test the training components
    import gym
    from preprocess import NoisyEnvWrapper, generate_demonstration_tokens
    
    # Create environment and policy
    env = NoisyEnvWrapper(gym.make("CartPole-v1"), noise_level=0.1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create BI-SDICL policy
    policy = create_policy(state_dim, action_dim, policy_type="bi_sdicl")
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # Generate demonstration tokens
    demo_tokens = generate_demonstration_tokens(seed=42)
    
    # Train for a few episodes
    print("Training BI-SDICL policy...")
    rewards = train_agent(env, policy, optimizer, num_episodes=5, demo_tokens=demo_tokens)
    print(f"Rewards: {rewards}")
    
    # Test saving and loading
    save_model(policy, "models/test_policy.pt")
    
    env.close()
