"""
ACAG-OVS Main Experiment Script

This script runs the ACAG-OVS experiments:
- Experiment 1: Adaptive Attention Calibration Ablation Study
- Experiment 2: Contrastive Token Alignment Effectiveness
- Experiment 3: Architecture-Agnostic Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse
from tqdm import tqdm

from preprocess import get_data_loader
from evaluate import run_experiment1, run_experiment3
from train import AdaptiveThresholding, DummySegmentationModel, DummyTokenModel, train_step, info_nce_loss, multi_view_token_extraction

def run_experiment2(data_loader, num_epochs=3, save_dir='logs'):
    """
    Run Experiment 2: Contrastive Token Alignment Effectiveness.
    
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader
        num_epochs (int): Number of training epochs
        save_dir (str): Directory to save results
        
    Returns:
        tuple: Standard losses, contrastive losses
    """
    print("\nRunning Experiment 2: Contrastive Token Alignment Effectiveness")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyTokenModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    losses_standard = []
    losses_contrast = []
    
    for epoch in range(num_epochs):
        epoch_loss_std = 0.0
        epoch_loss_contrast = 0.0
        num_batches = 0
        
        for images, masks in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            loss_std, _ = train_step(model, images, masks, optimizer, use_contrastive=False)
            loss_contrast, seg_loss_val = train_step(model, images, masks, optimizer, use_contrastive=True, alpha=1.0)
            
            epoch_loss_std += loss_std
            epoch_loss_contrast += loss_contrast
            num_batches += 1
        
        avg_loss_std = epoch_loss_std / num_batches
        avg_loss_contrast = epoch_loss_contrast / num_batches
        losses_standard.append(avg_loss_std)
        losses_contrast.append(avg_loss_contrast)
        
        print(f"Epoch {epoch+1}: Standard Loss={avg_loss_std:.4f}, Contrastive Loss={avg_loss_contrast:.4f}")
    
    epochs = np.arange(1, num_epochs+1)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, losses_standard, marker="o", label="Standard (No Contrastive)")
    plt.plot(epochs, losses_contrast, marker="o", label="Multi-view Contrastive")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Experiment 2: Training Losses (Segmentation + Contrastive)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "contrastive_training_loss.pdf"))
    print(f"Saved plot as '{os.path.join(save_dir, 'contrastive_training_loss.pdf')}'")
    plt.close()
    
    return losses_standard, losses_contrast

def test_all(data_dir='data', save_dir='logs', batch_size=2, num_samples=10, quick_test=True):
    """
    Run all ACAG-OVS experiments.
    
    Args:
        data_dir (str): Directory with input data
        save_dir (str): Directory to save results
        batch_size (int): Batch size for data loading
        num_samples (int): Number of samples in the dataset
        quick_test (bool): Whether to run a quick test with minimal data
    """
    print("\n" + "="*80)
    print("Starting ACAG-OVS experiments")
    print("="*80)
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    data_loader = get_data_loader(batch_size=batch_size, num_samples=num_samples, data_dir=data_dir)
    
    start_time = time.time()
    fixed_score, adaptive_score = run_experiment1(data_loader, save_dir=save_dir)
    print(f"Experiment 1 completed in {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    num_epochs = 2 if quick_test else 5
    losses_standard, losses_contrast = run_experiment2(data_loader, num_epochs=num_epochs, save_dir=save_dir)
    print(f"Experiment 2 completed in {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    score_A, score_B = run_experiment3(data_loader, save_dir=save_dir)
    print(f"Experiment 3 completed in {time.time() - start_time:.2f} seconds")
    
    model_path = os.path.join('models', 'acag_ovs_model.pt')
    dummy_model = DummySegmentationModel().to(device)
    torch.save(dummy_model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    print("\n" + "="*80)
    print("ACAG-OVS experiments completed successfully")
    print("="*80)
    
    return {
        'experiment1': {
            'fixed_score': fixed_score,
            'adaptive_score': adaptive_score
        },
        'experiment2': {
            'losses_standard': losses_standard,
            'losses_contrast': losses_contrast
        },
        'experiment3': {
            'score_A': score_A,
            'score_B': score_B
        }
    }

def main():
    """
    Main function to parse arguments and run experiments.
    """
    parser = argparse.ArgumentParser(description='ACAG-OVS Experiments')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory with input data')
    parser.add_argument('--save_dir', type=str, default='logs', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for data loading')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples in the dataset')
    parser.add_argument('--quick_test', action='store_true', help='Run a quick test with minimal data')
    args = parser.parse_args()
    
    results = test_all(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        quick_test=args.quick_test
    )
    
    print("\nResults Summary:")
    print(f"Experiment 1 - Fixed Score: {results['experiment1']['fixed_score']:.4f}, "
          f"Adaptive Score: {results['experiment1']['adaptive_score']:.4f}")
    print(f"Experiment 2 - Final Standard Loss: {results['experiment2']['losses_standard'][-1]:.4f}, "
          f"Final Contrastive Loss: {results['experiment2']['losses_contrast'][-1]:.4f}")
    print(f"Experiment 3 - Model A Score: {results['experiment3']['score_A']:.4f}, "
          f"Model B Score: {results['experiment3']['score_B']:.4f}")

if __name__ == '__main__':
    main()
