"""
Main module for MCAD experiments.
Orchestrates the entire experimental pipeline.
"""
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from preprocess import set_seed, load_dataset, create_dataloaders, subsample_dataset
from train import create_model, create_trainer, train_epoch
from evaluate import evaluate_model, visualize_reconstructions, compare_models

def load_config(config_path="config/mcad_config.yaml"):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def experiment1_convergence(config, train_loader, test_loader, device):
    """
    Experiment 1: Convergence Speed and Reconstruction Quality.
    Compares MCAD with Base method on training speed and quality.
    """
    print("\n" + "="*80)
    print("[Experiment 1] Convergence Speed and Reconstruction Quality")
    print("="*80)
    
    # Set random seed for reproducibility
    set_seed(config['seed'])
    
    # Create models
    print("Creating models...")
    model_base = create_model(config, device)
    model_mcad = create_model(config, device)
    
    # Create trainers with different configurations
    base_config = config.copy()
    base_config['model']['use_momentum'] = False
    base_config['model']['adaptive_consistency'] = False
    
    trainer_base = create_trainer(model_base, base_config, device)
    trainer_mcad = create_trainer(model_mcad, config, device)
    
    # Run experiment for specified number of epochs
    num_epochs = config['experiments']['convergence']['epochs']
    max_batches = None if not config['test_run']['enabled'] else config['test_run']['max_batches']
    
    print(f"Training for {num_epochs} epochs...")
    
    base_losses = []
    mcad_losses = []
    base_times = []
    mcad_times = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train Base model
        print("Training Base model...")
        base_results = train_epoch(trainer_base, train_loader, base_config, max_batches=max_batches)
        base_losses.append(base_results['avg_loss'])
        base_times.append(base_results['epoch_time'])
        
        # Train MCAD model
        print("Training MCAD model...")
        mcad_results = train_epoch(trainer_mcad, train_loader, config, max_batches=max_batches)
        mcad_losses.append(mcad_results['avg_loss'])
        mcad_times.append(mcad_results['epoch_time'])
        
        # Print comparison
        print(f"Epoch {epoch+1} results:")
        print(f"  Base: Loss = {base_results['avg_loss']:.4f}, Time = {base_results['epoch_time']:.2f}s")
        print(f"  MCAD: Loss = {mcad_results['avg_loss']:.4f}, Time = {mcad_results['epoch_time']:.2f}s")
    
    # Evaluate models on test set
    print("\nEvaluating models on test set...")
    models_dict = {
        'Base': model_base,
        'MCAD': model_mcad
    }
    eval_results = compare_models(models_dict, test_loader, device, max_batches=max_batches)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(base_losses, label='Base')
    plt.plot(mcad_losses, label='MCAD')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.savefig('logs/convergence_loss.png')
    plt.close()
    
    # Plot time per epoch
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(num_epochs) - 0.2, base_times, width=0.4, label='Base')
    plt.bar(np.arange(num_epochs) + 0.2, mcad_times, width=0.4, label='MCAD')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Training Time Comparison')
    plt.legend()
    plt.savefig('logs/convergence_time.png')
    plt.close()
    
    # Visualize reconstructions
    print("Generating reconstruction visualizations...")
    vis_path = visualize_reconstructions(model_mcad, test_loader, device)
    
    return {
        'base_losses': base_losses,
        'mcad_losses': mcad_losses,
        'base_times': base_times,
        'mcad_times': mcad_times,
        'eval_results': eval_results,
        'visualization': vis_path
    }

def experiment2_robustness(config, full_train_dataset, test_loader, device):
    """
    Experiment 2: Robustness Under Limited and Corrupted Data Regimes.
    Tests performance with varying amounts of training data and noise levels.
    """
    print("\n" + "="*80)
    print("[Experiment 2] Robustness Under Limited and Corrupted Data Regimes")
    print("="*80)
    
    # Get configuration for this experiment
    data_percentages = config['experiments']['robustness']['data_percentages']
    noise_levels = config['experiments']['robustness']['noise_levels']
    num_seeds = config['experiments']['robustness']['seeds']
    
    # Use smaller sets for test run
    if config['test_run']['enabled']:
        data_percentages = data_percentages[:2]
        noise_levels = noise_levels[:2]
        num_seeds = 1
        max_batches = config['test_run']['max_batches']
    else:
        max_batches = None
    
    print(f"Testing with data percentages: {data_percentages}")
    print(f"Testing with noise levels: {noise_levels}")
    print(f"Number of random seeds: {num_seeds}")
    
    # Store results
    results = {}
    
    # Loop over random seeds, percentages, and noise level
    for seed in range(num_seeds):
        set_seed(config['seed'] + seed)
        print(f"\nSeed {seed+1}/{num_seeds}:")
        
        for pct in data_percentages:
            # Create subset of training data
            print(f"  Creating dataset with {pct*100}% of training data...")
            sub_dataset = subsample_dataset(full_train_dataset, percentage=pct)
            sub_loader = torch.utils.data.DataLoader(
                sub_dataset, 
                batch_size=config['dataset']['batch_size'],
                shuffle=True
            )
            
            for noise_std in noise_levels:
                print(f"  Testing with noise level: {noise_std}")
                
                # Create models for this configuration
                model_base = create_model(config, device)
                model_mcad = create_model(config, device)
                
                # Create trainers
                base_config = config.copy()
                base_config['model']['use_momentum'] = False
                base_config['model']['adaptive_consistency'] = False
                base_config['training']['noise_std'] = noise_std
                
                mcad_config = config.copy()
                mcad_config['training']['noise_std'] = noise_std
                
                trainer_base = create_trainer(model_base, base_config, device)
                trainer_mcad = create_trainer(model_mcad, mcad_config, device)
                
                # Train for one epoch (or use convergence epochs if specified)
                num_epochs = 1 if config['test_run']['enabled'] else config['experiments']['convergence']['epochs']
                
                # Train both models
                base_loss = 0
                mcad_loss = 0
                
                for epoch in range(num_epochs):
                    # Train Base model
                    base_results = train_epoch(
                        trainer_base, 
                        sub_loader, 
                        base_config, 
                        max_batches=max_batches,
                        log=False
                    )
                    # Train MCAD model
                    mcad_results = train_epoch(
                        trainer_mcad, 
                        sub_loader, 
                        mcad_config, 
                        max_batches=max_batches,
                        log=False
                    )
                    
                    base_loss = base_results['avg_loss']
                    mcad_loss = mcad_results['avg_loss']
                
                # Evaluate on test set
                base_eval = evaluate_model(
                    model_base, 
                    test_loader, 
                    device, 
                    noise_std=noise_std,
                    max_batches=max_batches
                )
                
                mcad_eval = evaluate_model(
                    model_mcad, 
                    test_loader, 
                    device, 
                    noise_std=noise_std,
                    max_batches=max_batches
                )
                
                # Store results
                key = (seed, pct, noise_std)
                results[key] = {
                    'base': {
                        'train_loss': base_loss,
                        'test_loss': base_eval['loss'],
                        'test_psnr': base_eval['psnr']
                    },
                    'mcad': {
                        'train_loss': mcad_loss,
                        'test_loss': mcad_eval['loss'],
                        'test_psnr': mcad_eval['psnr']
                    }
                }
                
                # Print results
                print(f"  Results for {pct*100}% data, noise={noise_std}:")
                print(f"    Base: Train Loss={base_loss:.4f}, Test Loss={base_eval['loss']:.4f}, PSNR={base_eval['psnr']:.2f}")
                print(f"    MCAD: Train Loss={mcad_loss:.4f}, Test Loss={mcad_eval['loss']:.4f}, PSNR={mcad_eval['psnr']:.2f}")
    
    # Create summary plots
    # We'll create a grid of plots for different data percentages and noise levels
    if len(results) > 0:
        fig, axes = plt.subplots(
            len(data_percentages), 
            len(noise_levels), 
            figsize=(4*len(noise_levels), 4*len(data_percentages)),
            squeeze=False
        )
        
        for i, pct in enumerate(data_percentages):
            for j, noise in enumerate(noise_levels):
                # Calculate averages across seeds
                base_psnrs = []
                mcad_psnrs = []
                
                for seed in range(num_seeds):
                    key = (seed, pct, noise)
                    if key in results:
                        base_psnrs.append(results[key]['base']['test_psnr'])
                        mcad_psnrs.append(results[key]['mcad']['test_psnr'])
                
                # Plot PSNR comparison
                axes[i, j].bar(['Base', 'MCAD'], [np.mean(base_psnrs), np.mean(mcad_psnrs)])
                axes[i, j].set_title(f"Data: {pct*100}%, Noise: {noise}")
                axes[i, j].set_ylabel('PSNR (dB)')
        
        plt.tight_layout()
        plt.savefig('logs/robustness_psnr.png')
        plt.close()
    
    return results

def experiment3_ablation(config, train_loader, test_loader, device):
    """
    Experiment 3: Ablation Study on Momentum Term and Hybrid Consistency Loss.
    Tests different configurations of the MCAD method.
    """
    print("\n" + "="*80)
    print("[Experiment 3] Ablation Study on Momentum Term and Hybrid Consistency Loss")
    print("="*80)
    
    # Get configuration for this experiment
    variants = config['experiments']['ablation']['variants']
    
    # Use smaller sets for test run
    if config['test_run']['enabled']:
        variants = variants[:2]
        max_batches = config['test_run']['max_batches']
    else:
        max_batches = None
    
    print(f"Testing with variants: {[v['name'] for v in variants]}")
    
    # Store results
    results = {}
    
    # Loop over variants
    for variant in variants:
        print(f"\nTesting variant: {variant['name']}")
        
        # Create model and trainer for this variant
        model = create_model(config, device)
        variant_config = config.copy()
        variant_config['model']['use_momentum'] = variant['use_momentum']
        variant_config['model']['adaptive_consistency'] = variant['adaptive_consistency']
        
        trainer = create_trainer(model, variant_config, device)
        
        # Train for one epoch (or use convergence epochs if specified)
        num_epochs = 1 if config['test_run']['enabled'] else config['experiments']['convergence']['epochs']
        
        # Train model
        for epoch in range(num_epochs):
            train_epoch(
                trainer, 
                train_loader, 
                variant_config, 
                max_batches=max_batches,
                log=False
            )
        
        # Evaluate on test set
        eval_results = evaluate_model(
            model, 
            test_loader, 
            device, 
            noise_std=config['training']['noise_std'],
            max_batches=max_batches
        )
        
        # Store results
        results[variant['name']] = eval_results
        
        # Print results
        print(f"  {variant['name']}: Test Loss={eval_results['loss']:.4f}, PSNR={eval_results['psnr']:.2f}")
    
    return results

def run_tests():
    """
    Run quick tests to verify the implementation.
    """
    print("\n" + "="*80)
    print("Running quick tests to verify implementation")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(config)
    train_loader, test_loader = create_dataloaders(config, train_dataset, test_dataset)
    
    # Run experiments
    experiment1_convergence(config, train_loader, test_loader, device)
    experiment2_robustness(config, train_dataset, test_loader, device)
    experiment3_ablation(config, train_loader, test_loader, device)
    
    print("\nQuick tests completed.")

def main():
    """
    Main function to run the full experiments.
    """
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(config)
    train_loader, test_loader = create_dataloaders(config, train_dataset, test_dataset)
    
    # Run experiments
    if config['experiments']['convergence']['enabled']:
        experiment1_convergence(config, train_loader, test_loader, device)
    
    if config['experiments']['robustness']['enabled']:
        experiment2_robustness(config, train_dataset, test_loader, device)
    
    if config['experiments']['ablation']['enabled']:
        experiment3_ablation(config, train_loader, test_loader, device)
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    # Run quick tests for verification
    run_tests()
    
    # Uncomment the following line to run full experiments
    # main()
