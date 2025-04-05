"""
Utility functions for model training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import os

def train_model(model, dataloaders, optimizer, criterion, num_epochs=50, device='cuda'):
    """
    Train a model for a specified number of epochs.
    
    Args:
        model: Model to train
        dataloaders: Dictionary containing training and validation dataloaders
        optimizer: Optimizer to use for training
        criterion: Loss function
        num_epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        train_loss_history: List of training losses
        val_loss_history: List of validation losses
        train_acc_history: List of training accuracies
        val_acc_history: List of validation accuracies
    """
    model.to(device)
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    os.makedirs('logs', exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        start_time = time.time()
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        elapsed = time.time() - start_time
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {elapsed:.2f}s")
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc)
        
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history
