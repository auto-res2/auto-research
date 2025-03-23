#!/usr/bin/env python3
"""
Data preprocessing for HACP experiments.
This module handles loading data from the Gymnasium environments
and preparing it for model training and evaluation.
"""

import os
import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def create_environment(env_name="MiniGrid-Empty-8x8-v0"):
    """
    Create and return a Gymnasium environment.
    
    Args:
        env_name (str): Name of the Gymnasium environment.
    
    Returns:
        gym.Env: The created environment.
    """
    try:
        env = gym.make(env_name)
        return env
    except Exception as e:
        print(f"Error creating {env_name} environment: {e}")
        print("Falling back to CartPole-v1.")
        return gym.make('CartPole-v1')

def get_environment_info(env):
    """
    Extract important information from the environment.
    
    Args:
        env (gym.Env): The Gymnasium environment.
    
    Returns:
        tuple: (input_shape, output_size) where input_shape is the shape of 
               observations and output_size is the number of possible actions.
    """
    obs, _ = env.reset()
    input_shape = np.shape(obs)
    output_size = env.action_space.n
    return input_shape, output_size

def prepare_data_dir():
    """Ensure data directory exists."""
    os.makedirs(os.path.join("data"), exist_ok=True)
