"""
Main script for running SBDT experiments.
"""

import os
import torch
import argparse
from datetime import datetime
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_dataset, create_dataloaders
from src.experiments.sbdt_experiments import (
    experiment_robustness,
    experiment_ablation,
    experiment_anomaly_detection
)

# Import configuration
import config.sbdt_config as config

def main():
    """
    Main function to run SBDT experiments.
    """
    # Set up device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Create directories for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("logs", f"sbdt_experiments_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {config.DATASET_NAME} dataset...")
    dataset = load_dataset(
        config.DATASET_NAME,
        root=config.DATASET_PATH,
        subset_size=config.SUBSET_SIZE if config.TEST_RUN else None
    )
    
    # Create dataloader
    train_loader = create_dataloaders(dataset, batch_size=config.BATCH_SIZE)
    
    # Run experiments
    print("\n" + "="*50)
    print("Running SBDT Experiments")
    print("="*50 + "\n")
    
    # Experiment 1: Robustness Under Noise
    exp1_results = experiment_robustness(
        device, train_loader, config,
        save_dir=os.path.join(results_dir, "experiment1_robustness")
    )
    
    # Experiment 2: Parameter Ablation
    exp2_results = experiment_ablation(
        device, train_loader, config,
        save_dir=os.path.join(results_dir, "experiment2_ablation")
    )
    
    # Experiment 3: Anomaly Detection
    exp3_results = experiment_anomaly_detection(
        device, train_loader, config,
        save_dir=os.path.join(results_dir, "experiment3_anomaly")
    )
    
    # Print summary of results
    print("\n" + "="*50)
    print("SBDT Experiments Summary")
    print("="*50)
    print(f"Experiment 1 - Robustness: SBDT improvement = {exp1_results['improvement']:.4f}")
    print(f"Experiment 2 - Best ablation setting: {max(exp2_results.items(), key=lambda x: x[1])}")
    print(f"Experiment 3 - Stealth: Base vs SBDT anomaly scores = {exp3_results['mean_base_score']:.4f} vs {exp3_results['mean_sbdt_score']:.4f}")
    print("="*50 + "\n")
    
    print(f"All experiment results saved to {results_dir}")

if __name__ == "__main__":
    main()
