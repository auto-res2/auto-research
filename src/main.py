"""
Main module for the Purify-Tweedie++ experiment.

This script implements the Purify-Tweedie++ research experiment,
which integrates high-fidelity diffusion purification with a 
training and sampling scheme inspired by the "Consistent Diffusion
Meets Tweedie" approach.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from preprocess import DataManager
from train import create_model, train_model, PurifyTweediePlusPlus
from evaluate import (
    generate_adversaries, 
    evaluate_model, 
    experiment_ablation, 
    plot_ablation_results,
    experiment_robustness,
    plot_robustness_results,
    experiment_efficiency,
    plot_efficiency_results
)

import sys
sys.path.append(os.path.abspath("./config"))
try:
    from purify_tweedie_config import *
except ImportError:
    DATASET = "CIFAR10"
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    MODEL_TYPE = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    ENABLE_DOUBLE_TWEEDIE = True
    ENABLE_CONSISTENCY_LOSS = True
    ENABLE_ADAPTIVE_COV = True
    ATTACK_METHODS = ["FGSM", "PGD", "CW"]
    ATTACK_EPSILON = 8/255
    ATTACK_ALPHA = 2/255
    ATTACK_STEPS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_ID = 0
    SAVE_PDF = True
    PDF_DPI = 300
    PLOT_STYLE = "whitegrid"

def setup_environment():
    """
    Set up the environment for the experiment.
    
    Returns:
        str: Device to use for computation
    """
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    if torch.cuda.is_available():
        device = DEVICE
        torch.cuda.set_device(GPU_ID)
        print(f"Using GPU: {torch.cuda.get_device_name(GPU_ID)}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    sns.set_style(PLOT_STYLE)
    
    return device

def run_quick_test(model, test_loader, device):
    """
    Run a quick test to verify the model and experiment setup.
    
    Args:
        model (torch.nn.Module): Model to test
        test_loader (DataLoader): Test data loader
        device (str): Device to use
    """
    print("\n=== Running Quick Test ===")
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            for i in range(min(5, len(target))):
                print(f"Sample {i}: Predicted {predicted[i].item()}, Actual {target[i].item()}")
            
            pipeline = PurifyTweediePlusPlus(model=model, device=device)
            result = pipeline.purify(data)
            if isinstance(result, tuple) and len(result) == 3:
                purified, uncertainties, _ = result
            else:
                purified, uncertainties = result
            
            print(f"Purification test: Mean uncertainty = {uncertainties.mean().item():.4f}")
            break  # Only test on one batch
    
    print("Quick test completed successfully.")

def main():
    """
    Main function to run the Purify-Tweedie++ experiment.
    """
    print("=== Starting Purify-Tweedie++ Experiment ===")
    
    device = setup_environment()
    
    print("\nLoading data...")
    data_manager = DataManager(
        dataset=DATASET,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    train_loader, test_loader, classes = data_manager.load_data()
    print(f"Loaded {DATASET} dataset with {len(classes)} classes")
    
    print("\nCreating model...")
    model = create_model(
        model_name=MODEL_TYPE,
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED
    )
    model = model.to(device)
    print(f"Created {MODEL_TYPE} model with {NUM_CLASSES} output classes")
    
    
    run_quick_test(model, test_loader, device)
    
    print("\nGenerating adversarial examples for experiments...")
    adv_images, labels, clean_images = generate_adversaries(
        model, test_loader, 
        attack_name='PGD', 
        eps=ATTACK_EPSILON, 
        alpha=ATTACK_ALPHA, 
        steps=ATTACK_STEPS,
        device=device
    )
    print(f"Generated {len(adv_images)} adversarial examples")
    
    print("\n=== Experiment 1: Ablation Study of Novel Components ===")
    ablation_results = experiment_ablation(
        model, test_loader, clean_images, adv_images, labels, device
    )
    plot_ablation_results(ablation_results, output_dir="logs")
    
    robustness_results = experiment_robustness(model, test_loader, device)
    plot_robustness_results(robustness_results, output_dir="logs")
    
    efficiency_results = experiment_efficiency(model, test_loader, device)
    plot_efficiency_results(efficiency_results, output_dir="logs")
    
    print("\n=== Purify-Tweedie++ Experiment Completed ===")
    print("Results saved in the logs directory.")

if __name__ == "__main__":
    main()
