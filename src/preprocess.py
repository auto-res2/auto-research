"""
Data preprocessing module for ACM optimizer experiments.
"""

import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import json
import argparse
import sys

# Add the project root directory to the Python path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Try relative imports (when imported as a module)
    from .utils.experiment_utils import set_seed, get_device, ExperimentLogger
except ImportError:
    # Fall back to absolute imports (when run as a script)
    from utils.experiment_utils import set_seed, get_device, ExperimentLogger


class WikiTextDataset(Dataset):
    """
    Dataset for WikiText language modeling.
    
    Args:
        texts (list): List of text strings
        tokenizer: Tokenizer for encoding texts
        block_size (int): Size of text blocks
    """
    
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        for text in texts:
            tokenized = tokenizer.encode(text)
            # Partition text into blocks
            for i in range(0, len(tokenized) - block_size + 1, block_size):
                self.examples.append(tokenized[i:i+block_size])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def preprocess_synthetic_data(config):
    """
    Prepare synthetic function data.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary containing synthetic function data
    """
    try:
        # Try relative imports (when imported as a module)
        from .utils.synthetic_functions import create_random_psd_matrix, create_random_vector
    except ImportError:
        # Fall back to absolute imports (when run as a script)
        from utils.synthetic_functions import create_random_psd_matrix, create_random_vector
    
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Prepare data for quadratic function
    quadratic_dim = config["functions"]["quadratic"]["dim"]
    A_quad = create_random_psd_matrix(quadratic_dim, seed=config["random_seed"])
    b_quad = create_random_vector(quadratic_dim, seed=config["random_seed"])
    init_x_quad = np.random.randn(quadratic_dim)
    
    # Prepare data for Rosenbrock function
    init_x_rosen = np.array([-1.2, 1.0])
    
    # Prepare data for Rastrigin function
    rastrigin_dim = config["functions"]["rastrigin"]["dim"]
    init_x_rastrigin = np.random.uniform(-5.12, 5.12, size=rastrigin_dim)
    
    # Return all data
    return {
        "quadratic": {
            "init_x": init_x_quad,
            "A": A_quad,
            "b": b_quad
        },
        "rosenbrock": {
            "init_x": init_x_rosen,
            "a": config["functions"]["rosenbrock"]["a"],
            "b": config["functions"]["rosenbrock"]["b"]
        },
        "rastrigin": {
            "init_x": init_x_rastrigin,
            "A": config["functions"]["rastrigin"]["A"]
        }
    }


def preprocess_cifar10_data(config):
    """
    Preprocess CIFAR-10 dataset.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    data_dir = config["data"]["data_dir"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # If test_mode, use a small subset
    if config["test_mode"]:
        trainset.data = trainset.data[:500]
        testset.data = testset.data[:100]
    
    # Create data loaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def preprocess_transformer_data(config):
    """
    Preprocess data for transformer language modeling.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (dataloader, tokenizer)
    """
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # For demonstration, create a dummy dataset
    # In a real scenario, you would load actual WikiText data
    texts = ["This is an example sentence from WikiText-2 dataset."] * 1000
    
    # Create dataset
    block_size = config["data"]["block_size"] if not config["test_mode"] else 32
    
    # Ensure text is long enough for the block size
    if len(tokenizer.encode(texts[0])) < block_size:
        # Repeat the text to make it longer than block_size
        texts = [text * max(1, block_size // len(tokenizer.encode(text)) + 1) for text in texts]
    
    dataset = WikiTextDataset(texts, tokenizer, block_size=block_size)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"]
    )
    
    return dataloader, tokenizer


def preprocess_data(config_path, experiment_type):
    """
    Preprocess data based on experiment type.
    
    Args:
        config_path (str): Path to configuration file
        experiment_type (str): Type of experiment ('synthetic', 'cifar10', or 'transformer')
        
    Returns:
        tuple: Processed data and configuration
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create logger
    logger = ExperimentLogger(config["output_dir"], config["experiment_name"])
    logger.log("Starting data preprocessing...")
    logger.log_config(config)
    
    # Preprocess data based on experiment type
    if experiment_type == 'synthetic':
        data = preprocess_synthetic_data(config)
        logger.log("Synthetic data preprocessing completed.")
        return data, config
    
    elif experiment_type == 'cifar10':
        train_loader, test_loader = preprocess_cifar10_data(config)
        logger.log(f"CIFAR-10 data preprocessing completed. Train set size: {len(train_loader.dataset)}, Test set size: {len(test_loader.dataset)}")
        return (train_loader, test_loader), config
    
    elif experiment_type == 'transformer':
        dataloader, tokenizer = preprocess_transformer_data(config)
        logger.log(f"Transformer data preprocessing completed. Dataset size: {len(dataloader.dataset)}")
        return (dataloader, tokenizer), config
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing for ACM optimizer experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True, choices=['synthetic', 'cifar10', 'transformer'], help='Type of experiment')
    
    args = parser.parse_args()
    
    # Preprocess data
    data, config = preprocess_data(args.config, args.experiment)
    
    print(f"Data preprocessing for {args.experiment} experiment completed.")
