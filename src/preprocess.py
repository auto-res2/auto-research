#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements the preprocessing components for the BI-SDICL (Bias-Integrated Sequential 
Decision In-Context Learner) experiments, including environment wrappers for noise injection.
"""

import gym
import numpy as np
import random
import torch

class NoisyEnvWrapper(gym.Wrapper):
    """
    Environment wrapper that injects Gaussian noise into state transitions.
    Used to test robustness of BI-SDICL under different levels of environmental stochasticity.
    """
    def __init__(self, env, noise_level=0.0):
        super(NoisyEnvWrapper, self).__init__(env)
        self.noise_level = noise_level

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Inject noise into state transitions by adding Gaussian noise scaled with noise_level
        noisy_state = state + np.random.normal(0, self.noise_level, size=state.shape)
        return noisy_state, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def generate_demonstration_tokens(batch_size=1, num_demos=10, demo_embedding_dim=64, seed=None):
    """
    Generates synthetic demonstration tokens for the BI-SDICL model.
    
    Args:
        batch_size: Number of batches of demonstrations to generate
        num_demos: Number of demonstrations per batch
        demo_embedding_dim: Dimension of each demonstration embedding
        seed: Random seed for reproducibility
        
    Returns:
        torch.Tensor of shape (batch_size, num_demos, demo_embedding_dim)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate random demonstration tokens
    demo_tokens = torch.randn(batch_size, num_demos, demo_embedding_dim)
    return demo_tokens

def prepare_environment(env_name="CartPole-v1", noise_level=0.0):
    """
    Creates and prepares the environment for experiments.
    
    Args:
        env_name: Name of the Gym environment to create
        noise_level: Level of noise to inject into state transitions
        
    Returns:
        Wrapped environment instance
    """
    env = gym.make(env_name)
    if noise_level > 0:
        env = NoisyEnvWrapper(env, noise_level=noise_level)
    return env

def prepare_state_for_model(state):
    """
    Converts a state from the environment to the format expected by the model.
    
    Args:
        state: State from the environment
        
    Returns:
        Tensor representation of the state
    """
    return torch.FloatTensor(state).unsqueeze(0)  # shape: (1, state_dim)

if __name__ == "__main__":
    # Test the preprocessing components
    env = prepare_environment(noise_level=0.2)
    state = env.reset()
    print(f"Original state: {state}")
    
    state_tensor = prepare_state_for_model(state)
    print(f"State tensor shape: {state_tensor.shape}")
    
    demo_tokens = generate_demonstration_tokens(seed=42)
    print(f"Demo tokens shape: {demo_tokens.shape}")
    
    # Test the noisy environment
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    print(f"Next state after noise injection: {next_state}")
    
    env.close()
