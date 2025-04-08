"""
Evaluation script for DEALWGAN experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils.metrics import compute_fid, plot_tsne

def evaluate_model(model, test_loader, config):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model (DEALWGAN or LWGAN)
        test_loader: DataLoader for test data
        config: Configuration object
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    print(f"Evaluating {model.__class__.__name__}...")
    
    samples = model.generate_samples(num_samples=config.sample_size)
    
    fid = compute_fid(None, None)  # Placeholder for actual FID computation
    
    latent_reps = model.get_latent_representations(test_loader)
    
    plot_tsne(latent_reps)
    
    save_samples(samples, f"{model.__class__.__name__}_samples")
    
    metrics = {
        "fid": fid,
        "latent_dim": model.latent_dim
    }
    
    print(f"Evaluation complete. FID: {fid:.4f}")
    
    return metrics

def compare_models(models_dict, test_loader, config):
    """
    Compare multiple models on evaluation metrics.
    
    Args:
        models_dict: Dictionary of models to compare (name: model)
        test_loader: DataLoader for test data
        config: Configuration object
        
    Returns:
        comparison: Dictionary containing comparison results
    """
    print("Comparing models...")
    
    metrics_dict = {}
    samples_dict = {}
    
    for name, model in models_dict.items():
        metrics = evaluate_model(model, test_loader, config)
        metrics_dict[name] = metrics
        
        samples = model.generate_samples(num_samples=16)
        samples_dict[name] = samples
    
    compare_samples(samples_dict)
    
    fid_scores = {name: metrics["fid"] for name, metrics in metrics_dict.items()}
    plot_comparison_bar(fid_scores, "FID Score Comparison", "fid_comparison")
    
    comparison = {
        "metrics": metrics_dict,
        "fid_comparison": fid_scores
    }
    
    return comparison

def save_samples(samples, filename, nrow=8):
    """
    Save generated samples as a grid.
    
    Args:
        samples: Tensor of samples [N, C, H, W]
        filename: Output filename (will be saved as PDF)
        nrow: Number of images per row
    """
    samples_np = samples.detach().cpu().numpy()
    
    samples_np = np.transpose(samples_np, (0, 2, 3, 1))
    
    samples_np = (samples_np + 1) / 2.0
    
    n_samples = min(64, samples_np.shape[0])
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(samples_np[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"logs/{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved samples as logs/{filename}.pdf")

def plot_comparison_bar(data_dict, title, filename):
    """
    Create a bar plot comparing metrics across models.
    
    Args:
        data_dict: Dictionary of data to plot (name: value)
        title: Title for the plot
        filename: Output filename (will be saved as PDF)
    """
    plt.figure(figsize=(10, 6))
    
    names = list(data_dict.keys())
    values = list(data_dict.values())
    
    plt.bar(names, values)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(f"logs/{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot as logs/{filename}.pdf")

def compare_samples(samples_dict):
    """
    Create a visual comparison of samples from different models.
    
    Args:
        samples_dict: Dictionary of samples (name: samples)
    """
    plt.figure(figsize=(15, 10))
    
    n_models = len(samples_dict)
    n_samples = 4  # Number of samples to show per model
    
    for i, (name, samples) in enumerate(samples_dict.items()):
        samples_np = samples.detach().cpu().numpy()
        samples_np = np.transpose(samples_np, (0, 2, 3, 1))
        samples_np = (samples_np + 1) / 2.0
        
        for j in range(n_samples):
            plt.subplot(n_models, n_samples, i * n_samples + j + 1)
            plt.imshow(samples_np[j])
            if j == 0:
                plt.ylabel(name, fontsize=12)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("logs/sample_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved sample comparison as logs/sample_comparison.pdf")
