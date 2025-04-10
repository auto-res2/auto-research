"""
Dataset class for the DITTO-GSD experiments.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class Synthetic3DDataset(Dataset):
    """
    Synthetic 3D dataset for testing the DITTO-GSD model.
    
    Generates synthetic point clouds and ground-truth meshes.
    """
    def __init__(self, num_samples=100, num_points=1024):
        """
        Initialize the dataset.
        
        Args:
            num_samples: Number of samples in the dataset.
            num_points: Number of points in each point cloud.
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.point_clouds = [np.random.rand(num_points, 3).astype(np.float32) for _ in range(num_samples)]
        self.gt_meshes = [np.random.rand(num_points, 3).astype(np.float32) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pc = torch.tensor(self.point_clouds[idx], dtype=torch.float32)
        gt = torch.tensor(self.gt_meshes[idx], dtype=torch.float32)
        return pc, gt
