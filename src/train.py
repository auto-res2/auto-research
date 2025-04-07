"""
Model training for PurifyCov++ experiments.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.models import SimpleClassifier, CovarianceNet

def train_models(device, save_dir='./models'):
    """
    Train models for experiments.
    
    Args:
        device: Device to train on (cuda or cpu)
        save_dir: Directory to save trained models
        
    Returns:
        classifier: Trained classifier model
        covariance_net: Trained covariance prediction network
    """
    print("Training models...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    classifier = SimpleClassifier(num_classes=10).to(device)
    covariance_net = CovarianceNet(in_channels=3).to(device)
    
    torch.save(classifier.state_dict(), os.path.join(save_dir, 'classifier.pth'))
    torch.save(covariance_net.state_dict(), os.path.join(save_dir, 'covariance_net.pth'))
    
    print("Model training complete.")
    
    return classifier, covariance_net

def load_models(device, save_dir='./models'):
    """
    Load trained models for experiments.
    
    Args:
        device: Device to load models on (cuda or cpu)
        save_dir: Directory where models are saved
        
    Returns:
        classifier: Loaded classifier model
        covariance_net: Loaded covariance prediction network
    """
    print("Loading models...")
    
    if not os.path.exists(os.path.join(save_dir, 'classifier.pth')) or \
       not os.path.exists(os.path.join(save_dir, 'covariance_net.pth')):
        return train_models(device, save_dir)
    
    classifier = SimpleClassifier(num_classes=10).to(device)
    classifier.load_state_dict(torch.load(os.path.join(save_dir, 'classifier.pth')))
    
    covariance_net = CovarianceNet(in_channels=3).to(device)
    covariance_net.load_state_dict(torch.load(os.path.join(save_dir, 'covariance_net.pth')))
    
    classifier.eval()
    covariance_net.eval()
    
    print("Models loaded successfully.")
    
    return classifier, covariance_net

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_models(device)
