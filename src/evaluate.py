"""
Evaluation functions for SBDT experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.diffusion_utils import evaluate_reconstruction, plot_reconstructions

def evaluate_activation(model, dataloader, device="cpu", save_dir=None):
    """
    Evaluate model activation using SSIM between model output and original image.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to use for evaluation
        save_dir: Directory to save visualizations
        
    Returns:
        Mean SSIM score
    """
    model.eval()
    ssim_scores = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            # Assume the trigger prompt equals data (in practice, diffusion inversion is required)
            trigger = data
            recon = model(trigger)
            
            # Calculate SSIM for each image in batch
            for i in range(data.size(0)):
                score = evaluate_reconstruction(data[i], recon[i])
                ssim_scores.append(score)
            
            # Save visualizations if directory is provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plot_reconstructions(
                    data, recon, 
                    f"Model Reconstructions (Mean SSIM: {np.mean(ssim_scores):.4f})",
                    os.path.join(save_dir, "reconstructions.png")
                )
            
            break  # Process only one batch for quicker evaluation
    
    mean_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    return mean_ssim

def compute_reconstruction_errors(autoencoder, dataloader, device="cpu"):
    """
    Compute reconstruction errors for anomaly detection.
    
    Args:
        autoencoder: Trained autoencoder
        dataloader: DataLoader for evaluation
        device: Device to use for evaluation
        
    Returns:
        Numpy array of reconstruction errors
    """
    autoencoder.eval()
    errors = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon = autoencoder(data)
            batch_errors = ((data - recon)**2).mean(dim=[1,2,3]).cpu().numpy()
            errors.extend(batch_errors)
            break  # Process only one batch for quicker evaluation
    
    return np.array(errors)

def evaluate_anomaly_detection(clean_features, base_features, sbdt_features, save_dir=None):
    """
    Evaluate anomaly detection using Isolation Forest.
    
    Args:
        clean_features: Features from clean data
        base_features: Features from Base Method
        sbdt_features: Features from SBDT Method
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of evaluation results
    """
    # Train Isolation Forest on clean features
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(clean_features)
    
    # Compute anomaly scores
    base_scores = iso_forest.decision_function(base_features)
    sbdt_scores = iso_forest.decision_function(sbdt_features)
    
    # Save visualization if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.hist(base_scores, bins=30, alpha=0.5, label="Base Method Scores")
        plt.hist(sbdt_scores, bins=30, alpha=0.5, label="SBDT Scores")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Anomaly Score Distribution (Base vs. SBDT)")
        plt.savefig(os.path.join(save_dir, "anomaly_scores.png"))
        plt.close()
    
    # Calculate summary statistics
    results = {
        "mean_base_score": np.mean(base_scores),
        "mean_sbdt_score": np.mean(sbdt_scores),
        "std_base_score": np.std(base_scores),
        "std_sbdt_score": np.std(sbdt_scores),
    }
    
    return results
