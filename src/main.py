#!/usr/bin/env python

import os
import torch
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Import from our modules
from preprocess import PCFGDataset, pad_sequence, create_dataloaders
from train import TransformerDecoderModel, train_model, experiment_rule_extrapolation
from evaluate import experiment_latent_signature_stability, experiment_continual_adaptation

# Fix random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    # Set deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_experiment_directories():
    """Create necessary directories for experiment outputs"""
    directories = ['logs', 'models', 'config', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_experiment_config(args, filename='config/experiment_config.txt'):
    """Save experiment configuration to a file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Configuration saved to {filename}")

def run_full_experiment(args):
    """
    Run the complete DMI experiment with all three components:
    1. Synthetic Rule Extrapolation
    2. Latent Signature Stability
    3. Continual Adaptation
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*80)
    print(f"Starting Dynamic Memory-Integrated Identifiability (DMI) Experiment")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Create necessary directories
    create_experiment_directories()
    
    # Save experiment configuration
    save_experiment_config(args)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    print("\n" + "-"*80)
    print("EXPERIMENT 1: SYNTHETIC RULE EXTRAPOLATION")
    print("-"*80)
    
    # Run Experiment 1: Rule Extrapolation
    exp1_results = experiment_rule_extrapolation(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        save_models=True
    )
    
    baseline_model = exp1_results['baseline_model']
    dmi_model = exp1_results['dmi_model']
    
    # Print experiment 1 summary
    print("\nExperiment 1 Summary:")
    print(f"Baseline OOD Loss: {exp1_results['baseline_ood_loss']:.4f}")
    print(f"DMI OOD Loss: {exp1_results['dmi_ood_loss']:.4f}")
    print(f"Improvement: {exp1_results['baseline_ood_loss'] - exp1_results['dmi_ood_loss']:.4f}")
    
    # Create and save visualization for Experiment 1
    plt.figure(figsize=(10, 6))
    plt.bar(['Baseline', 'DMI'], 
            [exp1_results['baseline_ood_loss'], exp1_results['dmi_ood_loss']])
    plt.ylabel('OOD Loss')
    plt.title('OOD Performance Comparison')
    plt.savefig('logs/ood_performance.png')
    plt.close()
    
    print("\n" + "-"*80)
    print("EXPERIMENT 2: LATENT SIGNATURE STABILITY")
    print("-"*80)
    
    # Run Experiment 2: Latent Signature Stability
    exp2_results = experiment_latent_signature_stability(baseline_model, dmi_model)
    
    # Print experiment 2 summary
    print("\nExperiment 2 Summary:")
    if exp2_results['sims_baseline'] and exp2_results['sims_dmi']:
        print(f"Average Baseline Similarity: {np.mean(exp2_results['sims_baseline']):.4f}")
        print(f"Average DMI Similarity: {np.mean(exp2_results['sims_dmi']):.4f}")
        if exp2_results['p_value'] is not None:
            print(f"Statistical significance (p-value): {exp2_results['p_value']:.4f}")
    
    print("\n" + "-"*80)
    print("EXPERIMENT 3: CONTINUAL ADAPTATION")
    print("-"*80)
    
    # Run Experiment 3: Continual Adaptation
    exp3_results = experiment_continual_adaptation(
        dmi_model, 
        batch_size=args.batch_size,
        adaptation_epochs=args.adaptation_epochs
    )
    
    # Print experiment 3 summary
    print("\nExperiment 3 Summary:")
    print(f"Pre-Adaptation Loss: {exp3_results['pre_adapt_loss']:.4f}")
    print(f"Post-Adaptation Loss: {exp3_results['post_adapt_loss']:.4f}")
    print(f"Improvement: {exp3_results['improvement']:.4f}")
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - SUMMARY OF FINDINGS")
    print("="*80)
    print(f"1. Rule Extrapolation: DMI showed {exp1_results['baseline_ood_loss'] - exp1_results['dmi_ood_loss']:.4f} lower loss on OOD data")
    
    if exp2_results['sims_baseline'] and exp2_results['sims_dmi']:
        sim_diff = np.mean(exp2_results['sims_dmi']) - np.mean(exp2_results['sims_baseline'])
        print(f"2. Latent Stability: DMI showed {sim_diff:.4f} higher similarity between related examples")
    
    print(f"3. Continual Adaptation: DMI model improved by {exp3_results['improvement']:.4f} after adaptation")
    
    print("\nAll experiment outputs and visualizations saved to logs/ directory")
    print("Trained models saved to models/ directory")
    print("\nExperiment completed successfully!")

def test_experiments():
    """
    Run a quick test of all experiments with minimal epochs and data.
    This is useful for verifying that the code works before running a full experiment.
    """
    print("Starting short tests for all experiments...\n")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create necessary directories
    create_experiment_directories()
    
    # Experiment 1: Rule Extrapolation with minimal settings
    print("Testing Experiment 1: Rule Extrapolation")
    exp1_results = experiment_rule_extrapolation(max_epochs=1, batch_size=32, save_models=True)
    
    baseline_model = exp1_results['baseline_model']
    dmi_model = exp1_results['dmi_model']
    
    # Experiment 2: Latent Signature Stability
    print("Testing Experiment 2: Latent Signature Stability")
    experiment_latent_signature_stability(baseline_model, dmi_model)
    
    # Experiment 3: Continual Learning / Robustness Under Data Shift
    print("Testing Experiment 3: Continual Adaptation")
    experiment_continual_adaptation(dmi_model, batch_size=32, adaptation_epochs=1)
    
    print("\nAll experiments have been executed successfully. Test finished.\n")

def main():
    """Main entry point for the experiment"""
    parser = argparse.ArgumentParser(description='Dynamic Memory-Integrated Identifiability (DMI) Experiment')
    
    # Experiment configuration
    parser.add_argument('--test', action='store_true', help='Run a quick test of all experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--adaptation_epochs', type=int, default=2, help='Number of epochs for continual adaptation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run either test or full experiment
    if args.test:
        test_experiments()
    else:
        run_full_experiment(args)

if __name__ == "__main__":
    main()
