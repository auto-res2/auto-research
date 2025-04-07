"""
Model loading and training for Cov-Purify++ experiments.
"""
import torch
import torch.nn as nn
from torchvision import models
from src.utils.metrics import compute_robust_accuracy

def load_model(model_name="resnet18", pretrained=True, num_classes=10, device=None):
    """
    Load a model for the experiments.
    
    Args:
        model_name (str): Name of the model to load
        pretrained (bool): Whether to use pretrained weights
        num_classes (int): Number of output classes
        device (torch.device): Device to load the model on
        
    Returns:
        torch.nn.Module: The loaded model
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        
        if num_classes != 1000:  # ImageNet has 1000 classes
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if device is not None:
        model = model.to(device)
    
    return model

def train_model(model, train_loader, test_loader, num_epochs=10, device=None):
    """
    Train a model on the given dataset.
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (torch.utils.data.DataLoader): Training data loader
        test_loader (torch.utils.data.DataLoader): Test data loader
        num_epochs (int): Number of training epochs
        device (torch.device): Device to train on
        
    Returns:
        torch.nn.Module: The trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            if device is not None:
                inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        test_accuracy = evaluate_model(model, test_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Test Acc: {test_accuracy:.2f}%")
    
    return model

def evaluate_model(model, data_loader, device=None):
    """
    Evaluate a model on the given dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to evaluate on
        
    Returns:
        float: Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            if device is not None:
                inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy
