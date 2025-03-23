"""
Preprocessing script for HBFG-SE3 experiments.
"""
import os
import torch
import numpy as np
import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_environment(config):
    """Set up the environment based on configuration."""
    # Set random seeds for reproducibility
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    # Set up device
    device = torch.device(config['experiment']['device'] 
                         if torch.cuda.is_available() else 'cpu')
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('logs/runs', exist_ok=True)
    
    return device

def preprocess_data():
    """
    Preprocess data for experiments.
    For this implementation, we use synthetic data.
    """
    print("Preprocessing data...")
    # Load configs
    experiment_config = load_config('config/experiment_config.yaml')
    model_config = load_config('config/model_config.yaml')
    
    # Set up environment
    device = setup_environment(experiment_config)
    
    return experiment_config, model_config, device
