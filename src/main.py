"""
This script implements three experiments comparing the proposed DEALWGAN versus the baseline LWGAN.
• Experiment 1: Performance and Convergence Benchmark on CIFAR-10.
• Experiment 2: Ablation Study to Isolate Contributions.
• Experiment 3: Robustness and Stability Analysis Across Hyperparameters.
All plots are saved as .pdf files using the naming convention:
    <figure_topic>[_<condition>][_pairN].pdf

Required Libraries:
    torch, torchvision, tensorboard, numpy, matplotlib, scikit-learn
"""

import torch
import random
import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.dealwgan_config import DEALWGANConfig

from preprocess import get_dataset
from train import (
    experiment_performance_convergence,
    experiment_ablation_study,
    experiment_robustness_analysis,
    run_test
)

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def main():
    """Main function to run the experiments."""
    print("=" * 80)
    print("DEALWGAN Experiments")
    print("=" * 80)
    
    create_directories()
    
    config = DEALWGANConfig()
    
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    config.device = device
    
    print(f"\nLoading {config.dataset} dataset...")
    train_loader, test_loader = get_dataset(
        config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    print(f"Dataset loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    print("\nStarting experiments...")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running in test mode (reduced epochs)...")
        run_test(config, train_loader, test_loader)
        return
    
    print("\n" + "=" * 50)
    experiment_performance_convergence(config, train_loader, test_loader)
    
    print("\n" + "=" * 50)
    experiment_ablation_study(config, train_loader, test_loader)
    
    print("\n" + "=" * 50)
    experiment_robustness_analysis(config, train_loader, test_loader)
    
    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
