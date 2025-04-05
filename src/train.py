"""
Training script for SphericalShift Point Transformer (SSPT) experiments.

This script handles model training for SSPT and baseline models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

from src.utils.data_utils import ModelNet40Dataset, ShapeNetDataset, create_dataloaders
from src.utils.train_utils import train_model
from src.utils.eval_utils import plot_training_curves
from src.models import SSPTModel, PTv3Model, SSPTVariant

def train_sspt_model(datasets, config, device='cuda'):
    """
    Train the SSPT model.
    
    Args:
        datasets: Dictionary containing training and validation datasets
        config: Configuration object with training parameters
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        model: Trained SSPT model
        train_loss_history: List of training losses
        val_loss_history: List of validation losses
        train_acc_history: List of training accuracies
        val_acc_history: List of validation accuracies
    """
    print("Training SSPT model...")
    
    dataloaders = create_dataloaders(
        datasets['train'],
        datasets['val'],
        batch_size=config.BATCH_SIZE
    )
    
    model = SSPTModel(num_classes=config.NUM_CLASSES)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
        model,
        dataloaders,
        optimizer,
        criterion,
        num_epochs=config.NUM_EPOCHS,
        device=device
    )
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/sspt_model.pth')
    
    plot_training_curves(
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        filename='sspt_training_curves.pdf'
    )
    
    print("SSPT model training completed.")
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def train_baseline_model(datasets, config, device='cuda'):
    """
    Train the baseline PTv3 model.
    
    Args:
        datasets: Dictionary containing training and validation datasets
        config: Configuration object with training parameters
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        model: Trained baseline model
        train_loss_history: List of training losses
        val_loss_history: List of validation losses
        train_acc_history: List of training accuracies
        val_acc_history: List of validation accuracies
    """
    print("Training baseline PTv3 model...")
    
    dataloaders = create_dataloaders(
        datasets['train'],
        datasets['val'],
        batch_size=config.BATCH_SIZE
    )
    
    model = PTv3Model(num_classes=config.NUM_CLASSES)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
        model,
        dataloaders,
        optimizer,
        criterion,
        num_epochs=config.NUM_EPOCHS,
        device=device
    )
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/ptv3_model.pth')
    
    plot_training_curves(
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        filename='ptv3_training_curves.pdf'
    )
    
    print("Baseline PTv3 model training completed.")
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def train_ablation_variants(datasets, config, device='cuda'):
    """
    Train SSPT variants for ablation study.
    
    Args:
        datasets: Dictionary containing training and validation datasets
        config: Configuration object with training parameters
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        results: Dictionary containing results for each variant
    """
    print("Training SSPT variants for ablation study...")
    
    dataloaders = create_dataloaders(
        datasets['train'],
        datasets['val'],
        batch_size=config.BATCH_SIZE
    )
    
    variants = {
        "Variant A (No Spherical Projection)": {
            "use_spherical_projection": False,
            "use_shifted_attention": True,
            "use_dual_attention": True,
            "use_spherical_pos_enc": True
        },
        "Variant B (Fixed-window Attention)": {
            "use_spherical_projection": True,
            "use_shifted_attention": False,
            "use_dual_attention": True,
            "use_spherical_pos_enc": True
        },
        "Variant C (No Vector-based Correlation in Dual Attention)": {
            "use_spherical_projection": True,
            "use_shifted_attention": True,
            "use_dual_attention": False,
            "use_spherical_pos_enc": True
        },
        "Variant D (Relative Positional Encoding)": {
            "use_spherical_projection": True,
            "use_shifted_attention": True,
            "use_dual_attention": True,
            "use_spherical_pos_enc": False
        }
    }
    
    results = {}
    
    for variant_name, variant_config in variants.items():
        print(f"Training {variant_name}...")
        
        model = SSPTVariant(
            num_classes=config.NUM_CLASSES,
            use_spherical_projection=variant_config["use_spherical_projection"],
            use_shifted_attention=variant_config["use_shifted_attention"],
            use_dual_attention=variant_config["use_dual_attention"],
            use_spherical_pos_enc=variant_config["use_spherical_pos_enc"]
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
            model,
            dataloaders,
            optimizer,
            criterion,
            num_epochs=config.NUM_EPOCHS // 2,  # Use fewer epochs for ablation study
            device=device
        )
        
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/sspt_variant_{variant_name.split()[0].lower()}.pth')
        
        results[variant_name] = {
            "model": model,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history,
            "final_train_loss": train_loss_history[-1],
            "final_val_loss": val_loss_history[-1],
            "final_train_acc": train_acc_history[-1],
            "final_val_acc": val_acc_history[-1]
        }
        
        print(f"{variant_name} training completed.")
    
    plt.figure(figsize=(12, 8))
    
    names = list(results.keys())
    val_accs = [results[name]["final_val_acc"] * 100 for name in names]
    
    plt.bar(range(len(val_accs)), val_accs, tick_label=names)
    plt.xticks(rotation=15, ha='right')
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Ablation Study - Validation Accuracy Comparison")
    
    for i, v in enumerate(val_accs):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('logs/ablation_study_results.pdf', format='pdf', dpi=300)
    plt.close()
    
    print("Ablation study completed.")
    
    return results

if __name__ == "__main__":
    print("Testing training functions...")
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.sspt_config import *
    
    class Config:
        def __init__(self):
            self.NUM_CLASSES = 40
            self.BATCH_SIZE = 8
            self.NUM_EPOCHS = 2
            self.LEARNING_RATE = 0.001
    
    config = Config()
    
    train_dataset = ModelNet40Dataset(split='train', augment=True, num_samples=10)
    val_dataset = ModelNet40Dataset(split='val', augment=False, num_samples=5)
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_sspt_model(
        datasets,
        config,
        device=device
    )
    
    print("Training test completed successfully.")
