"""Training functions for RG-MDS experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os

def setup_feature_extractor(device):
    """
    Set up a pretrained model for feature extraction.
    
    Args:
        device (torch.device): Device to put the model on
        
    Returns:
        torch.nn.Module: Feature extractor model
    """
    pretrained_model = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
    feature_extractor.eval()  # set to evaluation mode
    return feature_extractor.to(device)

def create_reference_gallery(dataset, feature_extractor, transform, device, max_refs=20):
    """
    Create a reference gallery from a dataset.
    
    Args:
        dataset: Dataset to create gallery from
        feature_extractor: Model for feature extraction
        transform: Transforms to apply
        device: Device to run on
        max_refs: Maximum number of references
        
    Returns:
        list: List of reference images
    """
    reference_gallery = []
    for idx in range(min(max_refs, len(dataset))):
        img, _ = dataset[idx]
        from torchvision.transforms.functional import to_pil_image
        ref_img = to_pil_image(img)
        reference_gallery.append(ref_img)
    return reference_gallery
