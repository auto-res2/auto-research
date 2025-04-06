"""
ClusterCloak: Evaluation Module

This module contains functions for evaluating poisoning defenses and
analyzing clustering performance.
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from utils.transforms import apply_clustercloak, apply_metacloak
from utils.metrics import compute_clustering_metrics

def analyze_feature_clustering(model, dataloader, method="ClusterCloak", save_plots=True, 
                             layer_name='layer4'):
    """
    Analyze feature clustering properties using a surrogate model.
    
    Args:
        model: Pretrained model for feature extraction
        dataloader: DataLoader providing images
        method: Method to apply ('ClusterCloak' or 'MetaCloak')
        save_plots: Whether to save visualization plots
        layer_name: Name of layer to extract features from
        
    Returns:
        Dictionary containing clustering metrics and extracted features
    """
    model.eval()
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(layer_name))
    
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            device = next(model.parameters()).device
            batch = batch.float().to(device)
            
            if method == "ClusterCloak":
                batch = apply_clustercloak(batch)
            elif method == "MetaCloak":
                batch = apply_metacloak(batch)
                
            _ = model(batch)
            feat = activation[layer_name].view(batch.size(0), -1).cpu().numpy()
            features.append(feat)
    
    features = np.vstack(features) if features else np.array([])
    
    if len(features) > 0:
        dummy_labels = np.argmax(features, axis=1)
        metrics = compute_clustering_metrics(features, dummy_labels)
    else:
        metrics = {'silhouette': -1, 'davies_bouldin': -1}
        
    return {
        'features': features,
        'metrics': metrics
    }

class DummyIdentityClassifier(nn.Module):
    """Simple classifier for evaluating identity preservation."""
    def __init__(self, img_size=224, num_classes=10):
        super(DummyIdentityClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * img_size * img_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
