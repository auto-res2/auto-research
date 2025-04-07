"""
Main script for running Cov-Purify++ experiments.
"""
import os
import torch
import time
from torchvision import models

from src.preprocess import load_cifar10, setup_device, set_random_seed
from src.train import load_model
from src.evaluate import (
    experiment1_comparative_robustness,
    experiment2_adaptive_solver,
    experiment3_hyperparameter_sensitivity
)

def main():
    """
    Main function to run all experiments for the Cov-Purify++ method.
    """
    print("=" * 80)
    print("Starting Cov-Purify++ Experiments")
    print("=" * 80)
    
    os.makedirs("logs", exist_ok=True)
    
    set_random_seed(42)
    
    device = setup_device()
    
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(batch_size=16, num_workers=4, image_size=224)
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")
    
    print("\nLoading pre-trained ResNet18 model...")
    model = load_model(model_name="resnet18", pretrained=True, num_classes=10, device=device)
    model.eval()  # Set model to evaluation mode for the experiments
    print("Model loaded successfully")
    
    print("\n" + "=" * 50)
    results_exp1 = experiment1_comparative_robustness(model, test_loader, device)
    
    print("\n" + "=" * 50)
    results_exp2 = experiment2_adaptive_solver(device)
    
    print("\n" + "=" * 50)
    results_exp3 = experiment3_hyperparameter_sensitivity(device)
    
    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print("Results and figures have been saved to the 'logs' directory")
    print("=" * 80)
    
    return {
        'experiment1': results_exp1,
        'experiment2': results_exp2,
        'experiment3': results_exp3
    }

if __name__ == "__main__":
    start_time = time.time()
    results = main()
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
