"""
Data preprocessing module for HBFN experiments.
"""

import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import os

class TreeDataset(Dataset):
    """Dataset for tree-structured data."""
    def __init__(self, features_dict):
        self.keys = list(features_dict.keys())
        self.data = [features_dict[k] for k in self.keys]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

def generate_tree(num_nodes=50, feature_dim=10, seed=42):
    """Generate a random tree and node features."""
    np.random.seed(seed)
    G = nx.complete_graph(num_nodes)
    T = nx.minimum_spanning_tree(G)
    features = {node: np.random.randn(feature_dim).astype(np.float32) for node in T.nodes()}
    return T, features

def prepare_tree_data(config):
    """Prepare tree dataset for experiments 1 and 3."""
    tree_graph, features = generate_tree(
        num_nodes=config["num_nodes"], 
        feature_dim=config["feature_dim"],
        seed=config["seed"]
    )
    
    dataset = TreeDataset(features)
    data_loader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    return tree_graph, dataset, data_loader

def prepare_mnist_data(config, data_dir="./data"):
    """Prepare MNIST dataset for experiment 2."""
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    mnist_data = torchvision.datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    data_loader = DataLoader(
        mnist_data, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    return mnist_data, data_loader
