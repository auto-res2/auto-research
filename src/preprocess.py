"""
Data preprocessing module for FahDiff experiments.
"""
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import random


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in globals():
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def generate_synthetic_graph(n_nodes, graph_type="barabasi_albert", m=2, seed=None):
    """
    Generate a synthetic graph for experiments.
    
    Args:
        n_nodes: Number of nodes in the graph
        graph_type: Type of graph to generate (barabasi_albert, erdos_renyi, etc.)
        m: Parameter for Barabási-Albert model (number of edges to attach)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph object
    """
    if seed is not None:
        set_seed(seed)
        
    if n_nodes < 3:
        n_nodes = 3  # BA model requires n>=3 for m=2
        
    if graph_type == "barabasi_albert":
        G = nx.barabasi_albert_graph(n=n_nodes, m=m)
    elif graph_type == "erdos_renyi":
        prob = m * 2 / (n_nodes - 1)  # Approximate edge probability for similar density
        G = nx.erdos_renyi_graph(n=n_nodes, p=prob)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
        
    return G


def prepare_dataset(config):
    """
    Prepare synthetic graph datasets for experiments.
    
    Args:
        config: Configuration dictionary with dataset parameters
        
    Returns:
        Dictionary of datasets with different sizes
    """
    set_seed(config["seed"])
    datasets = {}
    
    for size in config["dataset_sizes"]:
        datasets[size] = generate_synthetic_graph(
            n_nodes=size, 
            seed=config["seed"]
        )
        
    return datasets
