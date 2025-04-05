"""
Evaluation script for SphericalShift Point Transformer (SSPT) experiments.

This script handles model evaluation for SSPT and baseline models.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

from src.utils.data_utils import ModelNet40Dataset, ShapeNetDataset, create_dataloaders, augment_point_cloud
from src.utils.eval_utils import evaluate_model, plot_comparison_bar
from src.models import SSPTModel, PTv3Model, SSPTVariant

def evaluate_sspt_model(model, datasets, device='cuda'):
    """
    Evaluate the SSPT model.
    
    Args:
        model: Trained SSPT model
        datasets: Dictionary containing test dataset
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        accuracy: Accuracy of the model on the test dataset
    """
    print("Evaluating SSPT model...")
    
    accuracy = evaluate_model(model, datasets['test'], num_augmentations=3, device=device)
    
    print(f"SSPT model accuracy: {accuracy * 100:.2f}%")
    
    return accuracy

def evaluate_baseline_model(model, datasets, device='cuda'):
    """
    Evaluate the baseline PTv3 model.
    
    Args:
        model: Trained baseline model
        datasets: Dictionary containing test dataset
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        accuracy: Accuracy of the model on the test dataset
    """
    print("Evaluating baseline PTv3 model...")
    
    accuracy = evaluate_model(model, datasets['test'], num_augmentations=3, device=device)
    
    print(f"Baseline PTv3 model accuracy: {accuracy * 100:.2f}%")
    
    return accuracy

def compare_models(sspt_accuracy, baseline_accuracy):
    """
    Compare SSPT and baseline models.
    
    Args:
        sspt_accuracy: Accuracy of the SSPT model
        baseline_accuracy: Accuracy of the baseline model
    """
    print("Comparing SSPT and baseline models...")
    
    sspt_accuracy_pct = sspt_accuracy * 100
    baseline_accuracy_pct = baseline_accuracy * 100
    
    accuracies = [sspt_accuracy_pct, baseline_accuracy_pct]
    labels = ['SSPT', 'PTv3 Baseline']
    
    plot_comparison_bar(
        accuracies,
        labels,
        "Model Accuracy Comparison",
        "model_comparison.pdf"
    )
    
    improvement = sspt_accuracy_pct - baseline_accuracy_pct
    
    print(f"SSPT model accuracy: {sspt_accuracy_pct:.2f}%")
    print(f"Baseline PTv3 model accuracy: {baseline_accuracy_pct:.2f}%")
    print(f"Improvement: {improvement:.2f}%")
    
    return improvement

def evaluate_robustness(model, datasets, device='cuda'):
    """
    Evaluate model robustness under different perturbations.
    
    Args:
        model: Trained model
        datasets: Dictionary containing test dataset
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        results: Dictionary containing robustness results
    """
    print("Evaluating model robustness...")
    
    perturbations = {
        'rotation': {'rotation': True, 'scaling': False, 'noise': False},
        'scaling': {'rotation': False, 'scaling': True, 'noise': False},
        'noise': {'rotation': False, 'scaling': False, 'noise': True},
        'all': {'rotation': True, 'scaling': True, 'noise': True}
    }
    
    results = {}
    
    for pert_name, pert_params in perturbations.items():
        print(f"Evaluating robustness under {pert_name} perturbation...")
        
        test_dataset = datasets['test']
        perturbed_data = []
        perturbed_labels = []
        
        for i in range(len(test_dataset)):
            points, label = test_dataset[i]
            points_np = points.numpy()
            
            perturbed_points = augment_point_cloud(
                points_np,
                rotation=pert_params['rotation'],
                scaling=pert_params['scaling'],
                noise=pert_params['noise']
            )
            
            perturbed_data.append(perturbed_points)
            perturbed_labels.append(label.item())
        
        correct = 0
        total = 0
        
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for i in range(len(perturbed_data)):
                points = torch.tensor(perturbed_data[i], dtype=torch.float32).unsqueeze(0).to(device)
                label = perturbed_labels[i]
                
                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)
                
                correct += (predicted.item() == label)
                total += 1
        
        accuracy = correct / total
        results[pert_name] = accuracy * 100
        
        print(f"Accuracy under {pert_name} perturbation: {accuracy * 100:.2f}%")
    
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    accuracies = [results[name] for name in names]
    
    bars = plt.bar(names, accuracies, color=['blue', 'green', 'red', 'purple'])
    
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness under Different Perturbations')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('logs/robustness_results.pdf', format='pdf', dpi=300)
    plt.close()
    
    print("Robustness evaluation completed.")
    
    return results

def evaluate_ablation_variants(variant_models, datasets, device='cuda'):
    """
    Evaluate SSPT variants for ablation study.
    
    Args:
        variant_models: Dictionary containing trained variant models
        datasets: Dictionary containing test dataset
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        results: Dictionary containing evaluation results for each variant
    """
    print("Evaluating SSPT variants for ablation study...")
    
    results = {}
    
    for variant_name, variant_data in variant_models.items():
        print(f"Evaluating {variant_name}...")
        
        model = variant_data['model']
        
        accuracy = evaluate_model(model, datasets['test'], num_augmentations=3, device=device)
        
        results[variant_name] = accuracy * 100
        
        print(f"{variant_name} accuracy: {accuracy * 100:.2f}%")
    
    plt.figure(figsize=(12, 8))
    
    names = list(results.keys())
    accuracies = [results[name] for name in names]
    
    bars = plt.bar(names, accuracies, color=['blue', 'green', 'red', 'orange'])
    
    plt.xticks(rotation=15, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Study - Test Accuracy Comparison')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('logs/ablation_study_test_results.pdf', format='pdf', dpi=300)
    plt.close()
    
    print("Ablation study evaluation completed.")
    
    return results

if __name__ == "__main__":
    print("Testing evaluation functions...")
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.sspt_config import *
    
    class Config:
        def __init__(self):
            self.NUM_CLASSES = 40
    
    config = Config()
    
    test_dataset = ModelNet40Dataset(split='test', augment=False, num_samples=10)
    
    datasets = {
        'test': test_dataset
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SSPTModel(num_classes=config.NUM_CLASSES).to(device)
    
    accuracy = evaluate_sspt_model(model, datasets, device=device)
    
    print("Evaluation test completed successfully.")
