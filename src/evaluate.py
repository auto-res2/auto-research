"""
Evaluation module for IDRR-GAR experiments.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import os

def evaluate_model(model, eval_loader, device, config):
    """
    Evaluate the model on the provided data.
    
    Args:
        model: Model to evaluate
        eval_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        config: Configuration parameters
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, gt_depth) in enumerate(eval_loader):
            images = images.to(device)
            gt_depth = gt_depth.to(device)
            
            pred_depth = model(images)
            
            loss_photo = F.l1_loss(pred_depth, gt_depth)
            
            total_loss += loss_photo.item()
    
    avg_loss = total_loss / len(eval_loader)
    
    metrics = {
        'avg_loss': avg_loss
    }
    
    return metrics


def visualize_sampling(image, samples, height, width, filename):
    """
    Visualize sampled regions on an image.
    
    Args:
        image: Image tensor (C, H, W)
        samples: Indices of sampled regions
        height: Image height
        width: Image width
        filename: Output filename
    """
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize to [0,1]
    else:
        img_np = image
    
    overlay = img_np.copy()
    
    for idx in samples:
        y = int(idx // width)
        x = int(idx % width)
        cv2.circle(overlay, (x, y), radius=2, color=(1, 0, 0), thickness=-1)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.title("Sampled Regions")
    plt.axis('off')
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sampling visualization saved as {filename}")


def save_loss_comparison(loss_base, loss_idrr, title, filename):
    """
    Plot and save loss comparison between base model and IDRR-GAR model.
    
    Args:
        loss_base: List of losses for base model
        loss_idrr: List of losses for IDRR-GAR model
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_base, label="Base Method")
    plt.plot(loss_idrr, label="IDRR-GAR")
    plt.xlabel("Iteration (batch updates)")
    plt.ylabel("Combined Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss comparison plot saved as {filename}")


def save_loss_iterations(loss_history, title, filename):
    """
    Plot and save loss over iterations.
    
    Args:
        loss_history: List of losses over iterations
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Iteration loss plot saved as {filename}")


def save_bar_comparison(values, labels, title, filename):
    """
    Create and save bar chart comparison.
    
    Args:
        values: List of values for bars (can be tensors or floats)
        labels: List of labels for bars
        title: Plot title
        filename: Output filename
    """
    processed_values = []
    for val in values:
        if isinstance(val, torch.Tensor):
            processed_values.append(val.detach().cpu().item())
        else:
            processed_values.append(float(val))
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, processed_values)  # Use processed values here
    plt.ylabel("Loss Value")
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    value_str = ", ".join([f"{v:.4f}" for v in processed_values])
    print(f"Bar chart saved as {filename}")
    print(f"  - Values: {value_str}")
    print(f"  - Labels: {', '.join(labels)}")
    print(f"  - Title: {title}")


def run_statistical_test(values1, values2, name1, name2):
    """
    Run paired t-test and print results.
    
    Args:
        values1: First set of values
        values2: Second set of values
        name1: Name for first set
        name2: Name for second set
    """
    t_stat, p_value = stats.ttest_rel(values1, values2)
    print(f"Paired t-test: {name1} vs {name2}")
    print(f"t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
    
    if p_value < 0.05:
        print(f"The difference between {name1} and {name2} is statistically significant (p < 0.05)")
    else:
        print(f"The difference between {name1} and {name2} is not statistically significant (p >= 0.05)")
