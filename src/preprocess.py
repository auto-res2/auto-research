import torch
from torch_geometric.datasets import Planetoid

def load_dataset(dataset_name='Cora', root='/tmp'):
    """
    Load and preprocess a graph dataset.
    
    Args:
        dataset_name: Name of the dataset (default: 'Cora')
        root: Root directory for data storage (default: '/tmp')
        
    Returns:
        Processed dataset
    """
    if dataset_name == 'Cora':
        dataset = Planetoid(root=f'{root}/{dataset_name}', name=dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet")
    
    return dataset

def create_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Create a data loader for the given dataset.
    
    Args:
        dataset: Graph dataset
        batch_size: Batch size for training (default: 32)
        shuffle: Whether to shuffle the data (default: True)
        
    Returns:
        DataLoader for the dataset
    """
    from torch_geometric.loader import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
