"""
Data preprocessing for NSRPP experiments.
"""

import os
import numpy as np
import torch
import torch.utils.data as data
import yaml
from src.utils.data_generation import create_dataset
from src.utils.models import ThetaDataset

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the config file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def prepare_experiment1_data(config):
    """
    Initialize data for Experiment 1.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        dict: Dictionary containing initialized data for Experiment 1
    """
    return {
        'init_theta': config['experiment1']['init_theta'],
        'delta': config['experiment1']['delta'],
        'learning_rate': config['experiment1']['learning_rate'],
        'num_iters': config['experiment1']['num_iters']
    }

def prepare_experiment2_data(config):
    """
    Prepare data for Experiment 2 (surrogate ablation study).
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (train_loader, val_dataset) for training and evaluating surrogates
    """
    n_samples = config['experiment2']['n_samples']
    thetas, risks = create_dataset(n_samples=n_samples)
    
    split = int(0.75 * n_samples)
    train_thetas, val_thetas = thetas[:split], thetas[split:]
    train_risks, val_risks = risks[:split], risks[split:]

    train_dataset = ThetaDataset(train_thetas, train_risks)
    val_dataset = ThetaDataset(val_thetas, val_risks)
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader, val_dataset

def prepare_experiment3_data(config):
    """
    Initialize data for Experiment 3.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        dict: Dictionary containing initialized data for Experiment 3
    """
    return {
        'init_theta': config['experiment3']['init_theta'],
        'learning_rate': config['experiment3']['learning_rate'],
        'num_iters': config['experiment3']['num_iters']
    }
