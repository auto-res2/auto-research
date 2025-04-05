"""
Data preprocessing for SphericalShift Point Transformer (SSPT) experiments.

This script handles data loading, preprocessing, and augmentation for
point cloud datasets used in SSPT experiments.
"""
import torch
import numpy as np
import os
import sys
from utils.data_utils import ModelNet40Dataset, ShapeNetDataset, augment_point_cloud

def load_modelnet40_data(num_train_samples=1000, num_val_samples=200, num_test_samples=200, num_points=1024):
    """
    Load ModelNet40 dataset for classification.
    
    Args:
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        num_test_samples: Number of test samples
        num_points: Number of points per sample
        
    Returns:
        datasets: Dictionary containing training, validation, and test datasets
    """
    print("Loading ModelNet40 dataset...")
    
    train_dataset = ModelNet40Dataset(
        split='train',
        augment=True,
        num_samples=num_train_samples,
        num_points=num_points
    )
    
    val_dataset = ModelNet40Dataset(
        split='val',
        augment=False,
        num_samples=num_val_samples,
        num_points=num_points
    )
    
    test_dataset = ModelNet40Dataset(
        split='test',
        augment=False,
        num_samples=num_test_samples,
        num_points=num_points
    )
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    print(f"ModelNet40 dataset loaded with {num_train_samples} training samples, "
          f"{num_val_samples} validation samples, and {num_test_samples} test samples.")
    
    return datasets

def load_shapenet_data(num_train_samples=500, num_val_samples=100, num_test_samples=100, num_points=2048):
    """
    Load ShapeNet dataset for segmentation.
    
    Args:
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        num_test_samples: Number of test samples
        num_points: Number of points per sample
        
    Returns:
        datasets: Dictionary containing training, validation, and test datasets
    """
    print("Loading ShapeNet dataset...")
    
    train_dataset = ShapeNetDataset(
        split='train',
        augment=True,
        num_samples=num_train_samples,
        num_points=num_points
    )
    
    val_dataset = ShapeNetDataset(
        split='val',
        augment=False,
        num_samples=num_val_samples,
        num_points=num_points
    )
    
    test_dataset = ShapeNetDataset(
        split='test',
        augment=False,
        num_samples=num_test_samples,
        num_points=num_points
    )
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    print(f"ShapeNet dataset loaded with {num_train_samples} training samples, "
          f"{num_val_samples} validation samples, and {num_test_samples} test samples.")
    
    return datasets

def normalize_point_cloud(points):
    """
    Normalize point cloud to have zero mean and unit variance.
    
    Args:
        points: Point cloud data of shape (N, 3)
        
    Returns:
        normalized_points: Normalized point cloud
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    m = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / m
    
    return points

def preprocess_point_cloud(points, num_points=1024):
    """
    Preprocess point cloud by normalizing and sampling a fixed number of points.
    
    Args:
        points: Point cloud data
        num_points: Number of points to sample
        
    Returns:
        processed_points: Preprocessed point cloud
    """
    points = normalize_point_cloud(points)
    
    if points.shape[0] > num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[indices]
    elif points.shape[0] < num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=True)
        points = points[indices]
    
    return points

if __name__ == "__main__":
    print("Testing preprocessing functions...")
    
    points = np.random.rand(2000, 3)
    
    processed_points = preprocess_point_cloud(points, num_points=1024)
    
    print(f"Original point cloud shape: {points.shape}")
    print(f"Processed point cloud shape: {processed_points.shape}")
    
    datasets = load_modelnet40_data(num_train_samples=10, num_val_samples=5, num_test_samples=5)
    
    sample_points, sample_label = datasets['train'][0]
    
    print(f"Sample points shape: {sample_points.shape}")
    print(f"Sample label: {sample_label}")
    
    print("Preprocessing test completed successfully.")
