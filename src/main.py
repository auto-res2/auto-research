"""Main script for running RG-MDS experiments."""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
import yaml
from tqdm import tqdm

from src.preprocess import load_dataset
from src.train import setup_feature_extractor, create_reference_gallery
from src.evaluate import experiment1, experiment2, experiment3, run_tests

def main(config_path):
    """
    Run all RG-MDS experiments.
    
    Args:
        config_path (str): Path to configuration file
    """
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Running experiments with configuration from {config_path}")
    print(f"Configuration: {config}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = load_dataset(
        name=config['dataset']['name'],
        max_samples=config['dataset']['max_samples'],
        data_dir=config['dataset']['data_dir']
    )
    
    feature_extractor = setup_feature_extractor(device)
    
    if config['run_tests']:
        test_passed = run_tests(dataset, feature_extractor, transform, device)
        if not test_passed:
            print("Tests failed. Aborting experiments.")
            return
    
    if config['experiments']['run_experiment1']:
        experiment1(
            dataset, 
            feature_extractor, 
            transform, 
            device,
            max_samples=config['experiments']['experiment1']['max_samples'],
            save_plot=config['experiments']['experiment1']['save_plot'],
            output_dir=config['output_dir']
        )
    
    if config['experiments']['run_experiment2']:
        experiment2(
            dataset,
            feature_extractor,
            transform,
            device,
            max_samples=config['experiments']['experiment2']['max_samples'],
            save_plot=config['experiments']['experiment2']['save_plot'],
            output_dir=config['output_dir']
        )
    
    if config['experiments']['run_experiment3']:
        experiment3(
            dataset,
            feature_extractor,
            transform,
            device,
            max_samples=config['experiments']['experiment3']['max_samples'],
            save_plot=config['experiments']['experiment3']['save_plot'],
            output_dir=config['output_dir']
        )
    
    print("All experiments completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RG-MDS experiments')
    parser.add_argument('--config', type=str, default='./config/experiment.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
