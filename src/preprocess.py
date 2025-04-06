"""
Data preprocessing for ProtoSurvPath experiments.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.protosurvpath_config import *

class TCGADataset(Dataset):
    """Dataset class for TCGA-like data with gene expression and histology images."""
    
    def __init__(self, gene_data, image_data, survival_time, event):
        """
        Initialize the dataset.
        
        Args:
            gene_data: Gene expression data [num_samples, num_genes]
            image_data: Histology image data [num_samples, channels, height, width]
            survival_time: Survival time data
            event: Event indicator (0: censored, 1: event)
        """
        self.gene_data = gene_data
        self.image_data = image_data
        self.survival_time = survival_time
        self.event = event
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.survival_time)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return {
            'gene': torch.tensor(self.gene_data[idx], dtype=torch.float32),
            'image': torch.tensor(self.image_data[idx], dtype=torch.float32),
            'time': self.survival_time[idx],
            'event': self.event[idx]
        }

def perform_gmm_clustering(features, n_components=NUM_PROTOTYPES):
    """
    Perform Gaussian Mixture Model clustering to extract prototypes.
    
    Args:
        features: Feature vectors to cluster
        n_components: Number of clusters/prototypes
        
    Returns:
        prototypes: Cluster centers as prototypes
        cluster_assignments: Cluster assignments for each sample
    """
    gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_SEED)
    cluster_assignments = gmm.fit_predict(features)
    return gmm.means_, cluster_assignments

def generate_dummy_data(n_samples=200, quick_test=QUICK_TEST):
    """
    Generate dummy TCGA-like data for development and testing.
    
    Args:
        n_samples: Number of samples to generate
        quick_test: Flag for quick test with reduced samples
        
    Returns:
        gene_data: Gene expression data
        image_data: Histology image data
        survival_time: Survival time data
        event: Event indicator
    """
    if quick_test:
        n_samples = min(50, n_samples)
    
    gene_data = np.random.rand(n_samples, GENE_INPUT_DIM)
    
    image_data = np.random.rand(n_samples, IMAGE_CHANNELS, *IMAGE_SIZE)
    
    survival_time = np.random.rand(n_samples) * 10
    event = np.random.randint(0, 2, n_samples)
    
    return gene_data, image_data, survival_time, event

def prepare_data(gene_data, image_data, survival_time, event, test_size=TEST_SIZE, batch_size=BATCH_SIZE):
    """
    Prepare data for training and evaluation.
    
    Args:
        gene_data: Gene expression data
        image_data: Histology image data
        survival_time: Survival time data
        event: Event indicator
        test_size: Proportion of data for testing
        batch_size: Batch size for DataLoader
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    X_train_genes, X_test_genes, X_train_images, X_test_images, y_train_time, y_test_time, y_train_event, y_test_event = \
        train_test_split(gene_data, image_data, survival_time, event, test_size=test_size, random_state=RANDOM_SEED)
    
    train_dataset = TCGADataset(X_train_genes, X_train_images, y_train_time, y_train_event)
    test_dataset = TCGADataset(X_test_genes, X_test_images, y_test_time, y_test_event)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_or_generate_data():
    """
    Load data from disk or generate dummy data if not available.
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    
    print("Generating dummy TCGA-like data...")
    gene_data, image_data, survival_time, event = generate_dummy_data()
    
    
    train_loader, test_loader = prepare_data(gene_data, image_data, survival_time, event)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_or_generate_data()
    print(f"Train loader size: {len(train_loader.dataset)} samples")
    print(f"Test loader size: {len(test_loader.dataset)} samples")
    
    batch = next(iter(train_loader))
    print("Sample batch:")
    print(f"Gene data shape: {batch['gene'].shape}")
    print(f"Image data shape: {batch['image'].shape}")
    print(f"Survival time: {batch['time']}")
    print(f"Event indicator: {batch['event']}")
