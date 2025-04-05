"""
Main script for SphericalShift Point Transformer (SSPT) experiments.

This script orchestrates the entire experiment pipeline, from data preprocessing
to model training and evaluation. It implements the SSPT method described in the
research paper and compares it with the baseline PTv3 model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_utils import ModelNet40Dataset, ShapeNetDataset, create_dataloaders
from src.utils.train_utils import train_model
from src.utils.eval_utils import plot_training_curves, plot_comparison_bar
from src.models import SSPTModel, PTv3Model, SSPTVariant
from src.preprocess import load_modelnet40_data, load_shapenet_data
from src.train import train_sspt_model, train_baseline_model, train_ablation_variants
from src.evaluate import (
    evaluate_sspt_model, evaluate_baseline_model, compare_models,
    evaluate_robustness, evaluate_ablation_variants
)

def experiment_end_to_end(config, device='cuda'):
    """
    Run end-to-end benchmark experiment comparing SSPT with baseline PTv3.
    
    Args:
        config: Configuration object with experiment parameters
        device: Device to run on ('cuda' or 'cpu')
    """
    print("\n" + "="*80)
    print("Running Experiment 1: End-to-End Benchmark on Standard Datasets (Classification)")
    print("="*80)
    
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    datasets = load_modelnet40_data(
        num_train_samples=100,  # Reduced for quick testing
        num_val_samples=20,
        num_test_samples=20,
        num_points=config.NUM_POINTS
    )
    
    print("\nTraining SSPT Model...")
    sspt_model, sspt_train_loss, sspt_val_loss, sspt_train_acc, sspt_val_acc = train_sspt_model(
        datasets,
        config,
        device=device
    )
    
    print("\nTraining PTv3 Baseline Model...")
    baseline_model, baseline_train_loss, baseline_val_loss, baseline_train_acc, baseline_val_acc = train_baseline_model(
        datasets,
        config,
        device=device
    )
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sspt_train_loss, 'b-', label='SSPT Train')
    plt.plot(sspt_val_loss, 'b--', label='SSPT Val')
    plt.plot(baseline_train_loss, 'r-', label='PTv3 Train')
    plt.plot(baseline_val_loss, 'r--', label='PTv3 Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(sspt_train_acc, 'b-', label='SSPT Train')
    plt.plot(sspt_val_acc, 'b--', label='SSPT Val')
    plt.plot(baseline_train_acc, 'r-', label='PTv3 Train')
    plt.plot(baseline_val_acc, 'r--', label='PTv3 Val')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    
    plt.tight_layout()
    plt.savefig('logs/training_comparison.pdf', format='pdf', dpi=300)
    plt.close()
    
    print("\nEvaluating SSPT Model...")
    sspt_accuracy = evaluate_sspt_model(sspt_model, datasets, device=device)
    
    print("\nEvaluating PTv3 Baseline Model...")
    baseline_accuracy = evaluate_baseline_model(baseline_model, datasets, device=device)
    
    print("\nComparing SSPT and PTv3 Baseline Models...")
    improvement = compare_models(sspt_accuracy, baseline_accuracy)
    
    print(f"\nExperiment 1 completed. SSPT improvement over baseline: {improvement:.2f}%")

def experiment_ablation(config, device='cuda'):
    """
    Run ablation study experiment to analyze the contribution of each SSPT component.
    
    Args:
        config: Configuration object with experiment parameters
        device: Device to run on ('cuda' or 'cpu')
    """
    print("\n" + "="*80)
    print("Running Experiment 2: Component Ablation Study")
    print("="*80)
    
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    datasets = load_modelnet40_data(
        num_train_samples=80,  # Reduced for quick testing
        num_val_samples=20,
        num_test_samples=20,
        num_points=config.NUM_POINTS
    )
    
    print("\nTraining SSPT variants for ablation study...")
    variant_results = train_ablation_variants(datasets, config, device=device)
    
    print("\nEvaluating SSPT variants...")
    eval_results = evaluate_ablation_variants(variant_results, datasets, device=device)
    
    print("\nExperiment 2 completed.")

def experiment_robustness(config, device='cuda'):
    """
    Run robustness evaluation experiment to test SSPT under different perturbations.
    
    Args:
        config: Configuration object with experiment parameters
        device: Device to run on ('cuda' or 'cpu')
    """
    print("\n" + "="*80)
    print("Running Experiment 3: Robustness Evaluation under Perturbations")
    print("="*80)
    
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    datasets = load_modelnet40_data(
        num_train_samples=80,  # Reduced for quick testing
        num_val_samples=20,
        num_test_samples=30,
        num_points=config.NUM_POINTS
    )
    
    if os.path.exists('models/sspt_model.pth') and os.path.exists('models/ptv3_model.pth'):
        print("\nLoading pre-trained models...")
        
        sspt_model = SSPTModel(num_classes=config.NUM_CLASSES).to(device)
        sspt_model.load_state_dict(torch.load('models/sspt_model.pth'))
        
        baseline_model = PTv3Model(num_classes=config.NUM_CLASSES).to(device)
        baseline_model.load_state_dict(torch.load('models/ptv3_model.pth'))
    else:
        print("\nTraining new models...")
        
        sspt_model, _, _, _, _ = train_sspt_model(datasets, config, device=device)
        
        baseline_model, _, _, _, _ = train_baseline_model(datasets, config, device=device)
    
    print("\nEvaluating SSPT model robustness...")
    sspt_robustness = evaluate_robustness(sspt_model, datasets, device=device)
    
    print("\nEvaluating PTv3 baseline model robustness...")
    baseline_robustness = evaluate_robustness(baseline_model, datasets, device=device)
    
    print("\nComparing robustness results...")
    
    plt.figure(figsize=(12, 6))
    
    perturbations = list(sspt_robustness.keys())
    sspt_accuracies = [sspt_robustness[p] for p in perturbations]
    baseline_accuracies = [baseline_robustness[p] for p in perturbations]
    
    x = np.arange(len(perturbations))
    width = 0.35
    
    plt.bar(x - width/2, sspt_accuracies, width, label='SSPT')
    plt.bar(x + width/2, baseline_accuracies, width, label='PTv3 Baseline')
    
    plt.xlabel('Perturbation Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness Comparison under Different Perturbations')
    plt.xticks(x, perturbations)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('logs/robustness_comparison.pdf', format='pdf', dpi=300)
    plt.close()
    
    print("\nExperiment 3 completed.")

def test():
    """
    Run a quick test of all experiments with reduced dataset sizes and epochs.
    """
    print("\n" + "="*80)
    print("Running Quick Test of All Experiments")
    print("="*80)
    
    class Config:
        def __init__(self):
            self.RANDOM_SEED = 42
            self.GPU_DEVICE = 0
            self.NUM_POINTS = 1024
            self.NUM_CLASSES = 40
            self.BATCH_SIZE = 8
            self.NUM_EPOCHS = 2
            self.LEARNING_RATE = 0.001
            self.USE_SPHERICAL_PROJECTION = True
            self.USE_SHIFTED_ATTENTION = True
            self.USE_DUAL_ATTENTION = True
            self.USE_SPHERICAL_POS_ENC = True
            self.RUN_ABLATION = True
            self.RUN_ROBUSTNESS = True
    
    config = Config()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('logs', exist_ok=True)
    
    experiment_end_to_end(config, device=device)
    experiment_ablation(config, device=device)
    experiment_robustness(config, device=device)
    
    print("\nQuick test completed successfully.")

def main():
    """
    Main function to run the SSPT experiments.
    """
    parser = argparse.ArgumentParser(description='Run SSPT experiments')
    parser.add_argument('--test', action='store_true', help='Run a quick test of all experiments')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], help='Run a specific experiment (1: End-to-End, 2: Ablation, 3: Robustness)')
    parser.add_argument('--config', type=str, default='config/sspt_config.py', help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config.sspt_config import *
        
        class Config:
            def __init__(self):
                self.RANDOM_SEED = RANDOM_SEED
                self.GPU_DEVICE = GPU_DEVICE
                self.NUM_POINTS = NUM_POINTS
                self.NUM_CLASSES = NUM_CLASSES
                self.BATCH_SIZE = BATCH_SIZE
                self.NUM_EPOCHS = NUM_EPOCHS
                self.LEARNING_RATE = LEARNING_RATE
                self.USE_SPHERICAL_PROJECTION = USE_SPHERICAL_PROJECTION
                self.USE_SHIFTED_ATTENTION = USE_SHIFTED_ATTENTION
                self.USE_DUAL_ATTENTION = USE_DUAL_ATTENTION
                self.USE_SPHERICAL_POS_ENC = USE_SPHERICAL_POS_ENC
                self.RUN_ABLATION = RUN_ABLATION
                self.RUN_ROBUSTNESS = RUN_ROBUSTNESS
        
        config = Config()
    except ImportError:
        print("Configuration file not found. Using default configuration.")
        
        class Config:
            def __init__(self):
                self.RANDOM_SEED = 42
                self.GPU_DEVICE = 0
                self.NUM_POINTS = 1024
                self.NUM_CLASSES = 40
                self.BATCH_SIZE = 16
                self.NUM_EPOCHS = 50
                self.LEARNING_RATE = 0.001
                self.USE_SPHERICAL_PROJECTION = True
                self.USE_SHIFTED_ATTENTION = True
                self.USE_DUAL_ATTENTION = True
                self.USE_SPHERICAL_POS_ENC = True
                self.RUN_ABLATION = True
                self.RUN_ROBUSTNESS = True
        
        config = Config()
    
    device = torch.device(f"cuda:{config.GPU_DEVICE}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs('logs', exist_ok=True)
    
    if args.test:
        test()
    elif args.experiment == 1:
        experiment_end_to_end(config, device=device)
    elif args.experiment == 2:
        experiment_ablation(config, device=device)
    elif args.experiment == 3:
        experiment_robustness(config, device=device)
    else:
        experiment_end_to_end(config, device=device)
        
        if config.RUN_ABLATION:
            experiment_ablation(config, device=device)
        
        if config.RUN_ROBUSTNESS:
            experiment_robustness(config, device=device)
    
    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
