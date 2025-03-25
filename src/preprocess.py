import os
import torch
import numpy as np
from utils.utils import load_config, get_dataloader, set_seed
import argparse

def preprocess_data(config_path):
    """
    Preprocess data based on configuration.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        train_loader, test_loader: Data loaders for training and testing
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Get data loaders
    train_loader, test_loader = get_dataloader(config)
    
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Check one batch to ensure correct format
    data_sample, _ = next(iter(train_loader))
    print(f"Data shape: {data_sample.shape}")
    print(f"Data type: {data_sample.dtype}")
    print(f"Data range: [{data_sample.min():.4f}, {data_sample.max():.4f}]")
    
    return train_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing for TwiST-Distill")
    parser.add_argument("--config", type=str, default="config/twist_distill_config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()
    
    train_loader, test_loader = preprocess_data(args.config)
    print("Preprocessing completed successfully.")
