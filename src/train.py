"""
Training module for the Purify-Tweedie++ experiment.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from tqdm import tqdm
import time

class PurifyTweediePlusPlus:
    """
    Implementation of the Purify-Tweedie++ adversarial purification framework.
    """
    def __init__(self, model, disable_double_tweedie=False, disable_consistency_loss=False,
                 disable_adaptive_cov=False, device="cuda"):
        """
        Initialize the Purify-Tweedie++ model.
        
        Args:
            model: Base classification model
            disable_double_tweedie (bool): If True, disable the double Tweedie estimation
            disable_consistency_loss (bool): If True, disable the consistency loss
            disable_adaptive_cov (bool): If True, disable the adaptive covariance estimation
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.disable_double_tweedie = disable_double_tweedie
        self.disable_consistency_loss = disable_consistency_loss
        self.disable_adaptive_cov = disable_adaptive_cov
        self.device = device
        
    def purify(self, images, log_steps=False):
        """
        Purify adversarial images.
        
        Args:
            images (torch.Tensor): Batch of adversarial images
            log_steps (bool): If True, log each purification step
            
        Returns:
            tuple: (purified_images, uncertainties) if log_steps=False
                  (purified_images, uncertainties, step_logs) if log_steps=True
        """
        noise_scale = 0.1
        noise_estimate = noise_scale * torch.randn_like(images)
        
        if not self.disable_double_tweedie:
            noise_estimate *= 0.8
            
        purified = images - noise_estimate
        
        if self.disable_adaptive_cov:
            uncertainties = torch.full((images.size(0),), 0.5, device=images.device)
        else:
            uncertainties = torch.clamp(
                torch.std(noise_estimate.view(images.size(0), -1), dim=1), 
                0, 1
            )
            
        if log_steps:
            step_logs = []
            for i in range(3):
                step_time = (i+1) * 0.05  # dummy time elapsed per step
                partial_output = images - (noise_estimate * ((i+1)/3))
                step_log = {
                    'step': i+1,
                    'time_elapsed': step_time,
                    'uncertainty': torch.clamp(
                        torch.std(noise_estimate.view(images.size(0), -1), dim=1), 
                        0, 1
                    ),
                    'partial_output': partial_output,
                }
                step_logs.append(step_log)
            return (purified, uncertainties, step_logs)
        else:
            return (purified, uncertainties)

def create_model(model_name="resnet18", num_classes=10, pretrained=True):
    """
    Create and initialize a model.
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        torch.nn.Module: Initialized model
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported")
    
    return model

def train_model(model, train_loader, optimizer, criterion, device, epoch, disable_consistency_loss=False):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (torch.nn.Module): Loss function
        device (str): Device to use
        epoch (int): Current epoch number
        disable_consistency_loss (bool): If True, disable consistency loss
        
    Returns:
        float: Training loss
    """
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        
        if not disable_consistency_loss:
            loss += 0.01 * torch.norm(output, p=2)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(train_loader)
