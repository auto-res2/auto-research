"""
Preprocessing module for the Purify-Tweedie++ experiment.
"""

import os
import torch
from torchvision import datasets, transforms
import numpy as np

class DataManager:
    """
    Data manager class for loading and preprocessing datasets for the experiment.
    """
    def __init__(self, dataset="CIFAR10", batch_size=128, num_workers=4, data_dir="./data"):
        """
        Initialize the data manager.
        
        Args:
            dataset (str): Name of the dataset to use ('CIFAR10' is the only supported dataset)
            batch_size (int): Batch size for data loading
            num_workers (int): Number of worker threads for data loading
            data_dir (str): Directory to store the datasets
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def load_data(self):
        """
        Load the dataset.
        
        Returns:
            tuple: (train_loader, test_loader, classes)
        """
        if self.dataset == "CIFAR10":
            train_dataset = datasets.CIFAR10(
                root=self.data_dir, 
                train=True, 
                download=True, 
                transform=self.transform
            )
            
            test_dataset = datasets.CIFAR10(
                root=self.data_dir, 
                train=False, 
                download=True, 
                transform=self.transform
            )
            
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            raise ValueError(f"Dataset {self.dataset} is not supported")
            
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        return train_loader, test_loader, classes
