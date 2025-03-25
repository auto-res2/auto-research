#!/usr/bin/env python3
"""
Model evaluation module for CGCD experiments.

This module implements:
1. Evaluation metrics for diffusion models
2. Visualization of model outputs
3. Comparison between Base and CGCD models
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils.diffusion_utils import (
    seed_everything, get_device, load_model, 
    create_tensorboard_writer, plot_images, get_condition
)
from train import (
    BaseDiffusionModel, CGCDDiffusionModel, 
    DiffusionModelVariant, ContinuousCGCDModel
)

def evaluate_reconstruction(model, dataloader, model_name='Base', 
                           max_batches=None, device=None, 
                           log_dir='logs', save_images=True):
    """
    Evaluate model reconstruction quality.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        model_name: Name of the model
        max_batches: Maximum number of batches to evaluate
        device: Device to evaluate on
        log_dir: Directory for logs and visualizations
        save_images: Whether to save output images
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if device is None:
        device = get_device()
        
    model.to(device)
    model.eval()
    
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name} Reconstruction")
    print(f"{'='*50}")
    
    # Create log directories
    writer = create_tensorboard_writer(f'{log_dir}/{model_name}_eval')
    
    # Metrics
    total_mse = 0.0
    total_psnr = 0.0
    total_samples = 0
    
    # For visualization
    all_inputs = []
    all_outputs = []
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
                
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            condition = get_condition(labels).to(device)
            noise_level = 0.2
            alpha_t = torch.tensor(1.0).to(device)
            
            if model_name == 'CGCD' or 'Variant' in model_name or model_name == 'ContinuousCGCD':
                output, dynamic_alpha = model(data, condition, noise_level, alpha_t)
                print(f"Batch {i+1} - Dynamic Alpha: {dynamic_alpha.mean().item():.4f}")
            else:
                output = model(data, condition, noise_level, alpha_t)
            
            # Calculate metrics
            mse = nn.MSELoss()(output, data).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
            
            total_mse += mse * data.size(0)
            total_psnr += psnr * data.size(0)
            total_samples += data.size(0)
            
            # Store for visualization
            if i == 0 and save_images:
                all_inputs.append(data[:8].cpu())
                all_outputs.append(output[:8].cpu())
                
            print(f"Batch {i+1} - MSE: {mse:.4f}, PSNR: {psnr:.2f} dB")
    
    # Calculate average metrics
    avg_mse = total_mse / total_samples
    avg_psnr = total_psnr / total_samples
    
    print(f"\nEvaluation Results for {model_name}:")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Log metrics
    writer.add_scalar("Eval/MSE", avg_mse, 0)
    writer.add_scalar("Eval/PSNR", avg_psnr, 0)
    
    # Visualize results
    if save_images and all_inputs and all_outputs:
        all_inputs = torch.cat(all_inputs, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        
        # Create comparison visualization
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
        for i in range(min(8, all_inputs.size(0))):
            # Original image
            ax = axes[i*2//4, (i*2)%4]
            img = all_inputs[i].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Denormalize
            ax.imshow(img)
            ax.set_title("Original")
            ax.axis('off')
            
            # Reconstructed image
            ax = axes[(i*2+1)//4, (i*2+1)%4]
            img = all_outputs[i].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Denormalize
            ax.imshow(img)
            ax.set_title("Reconstructed")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{log_dir}/{model_name}_reconstruction.png")
        plt.close()
        
        print(f"Visualization saved to {log_dir}/{model_name}_reconstruction.png")
    
    writer.close()
    
    metrics = {
        'mse': avg_mse,
        'psnr': avg_psnr
    }
    
    print(f"{'='*50}\n")
    return metrics

def evaluate_experiment1(base_model, cgcd_model, dataloader, 
                        max_batches=None, device=None, log_dir='logs'):
    """
    Evaluate and compare Base and CGCD models (Experiment 1).
    
    Args:
        base_model: Base diffusion model
        cgcd_model: CGCD diffusion model
        dataloader: DataLoader for evaluation data
        max_batches: Maximum number of batches to evaluate
        device: Device to evaluate on
        log_dir: Directory for logs and visualizations
        
    Returns:
        comparison: Dictionary of comparison results
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: PERFORMANCE COMPARISON ON NOISY AND COMPLEX DATA")
    print("="*70)
    
    # Evaluate Base model
    base_metrics = evaluate_reconstruction(
        base_model, dataloader, model_name='Base',
        max_batches=max_batches, device=device, log_dir=log_dir
    )
    
    # Evaluate CGCD model
    cgcd_metrics = evaluate_reconstruction(
        cgcd_model, dataloader, model_name='CGCD',
        max_batches=max_batches, device=device, log_dir=log_dir
    )
    
    # Compare results
    mse_improvement = base_metrics['mse'] - cgcd_metrics['mse']
    psnr_improvement = cgcd_metrics['psnr'] - base_metrics['psnr']
    
    print("\nEXPERIMENT 1 COMPARISON:")
    print(f"MSE Improvement: {mse_improvement:.4f} ({mse_improvement/base_metrics['mse']*100:.2f}%)")
    print(f"PSNR Improvement: {psnr_improvement:.2f} dB")
    
    if mse_improvement > 0:
        print("CONCLUSION: CGCD outperforms Base model in reconstruction quality")
    else:
        print("CONCLUSION: Base model outperforms CGCD in reconstruction quality")
    
    comparison = {
        'base_metrics': base_metrics,
        'cgcd_metrics': cgcd_metrics,
        'mse_improvement': mse_improvement,
        'psnr_improvement': psnr_improvement
    }
    
    return comparison

def evaluate_experiment2(variants, variant_names, dataloader, 
                        max_batches=None, device=None, log_dir='logs'):
    """
    Evaluate model variants for ablation study (Experiment 2).
    
    Args:
        variants: List of model variants
        variant_names: List of variant names
        dataloader: DataLoader for evaluation data
        max_batches: Maximum number of batches to evaluate
        device: Device to evaluate on
        log_dir: Directory for logs and visualizations
        
    Returns:
        ablation_results: Dictionary of ablation study results
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: ABLATION STUDY OF ADAPTIVE COMPONENTS")
    print("="*70)
    
    all_metrics = {}
    
    # Evaluate each variant
    for model, name in zip(variants, variant_names):
        metrics = evaluate_reconstruction(
            model, dataloader, model_name=name,
            max_batches=max_batches, device=device, log_dir=log_dir
        )
        all_metrics[name] = metrics
    
    # Compare variants
    print("\nEXPERIMENT 2 COMPARISON:")
    for name, metrics in all_metrics.items():
        print(f"{name}: MSE = {metrics['mse']:.4f}, PSNR = {metrics['psnr']:.2f} dB")
    
    # Find best variant
    best_variant = min(all_metrics.items(), key=lambda x: x[1]['mse'])
    print(f"\nBest Variant: {best_variant[0]} with MSE = {best_variant[1]['mse']:.4f}")
    
    # Analyze component importance
    print("\nComponent Importance Analysis:")
    
    if 'Variant_Full' in all_metrics and 'Variant_Fixed' in all_metrics:
        adaptive_noise_impact = all_metrics['Variant_Fixed']['mse'] - all_metrics['Variant_Full']['mse']
        print(f"Adaptive Noise Impact: {adaptive_noise_impact:.4f} MSE reduction")
    
    if 'Variant_Full' in all_metrics and 'Variant_Hard' in all_metrics:
        soft_assimilation_impact = all_metrics['Variant_Hard']['mse'] - all_metrics['Variant_Full']['mse']
        print(f"Soft Assimilation Impact: {soft_assimilation_impact:.4f} MSE reduction")
    
    print("\nCONCLUSION:")
    if 'Variant_Full' in all_metrics and 'Variant_Fixed' in all_metrics and 'Variant_Hard' in all_metrics:
        # Compare the impact values if they exist
        if adaptive_noise_impact is not None and soft_assimilation_impact is not None:
            if float(adaptive_noise_impact) > float(soft_assimilation_impact):
                print("Adaptive noise scheduling has a greater impact on performance than soft assimilation")
            else:
                print("Soft assimilation has a greater impact on performance than adaptive noise scheduling")
    
    return all_metrics

def evaluate_experiment3(continuous_model, dataloader, 
                        max_batches=None, device=None, log_dir='logs'):
    """
    Evaluate continuous conditioning model (Experiment 3).
    
    Args:
        continuous_model: Continuous CGCD model
        dataloader: DataLoader for continuous data
        max_batches: Maximum number of batches to evaluate
        device: Device to evaluate on
        log_dir: Directory for logs and visualizations
        
    Returns:
        continuous_metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: CONTINUOUS FEATURE DOMAINS WITH EXTERNAL CUES")
    print("="*70)
    
    if device is None:
        device = get_device()
        
    continuous_model.to(device)
    continuous_model.eval()
    
    # Create log directories
    writer = create_tensorboard_writer(f'{log_dir}/ContinuousCGCD_eval')
    
    # Metrics
    total_mse = 0.0
    total_samples = 0
    
    # For signal correlation analysis
    all_signals = []
    all_latent_means = []
    
    with torch.no_grad():
        for i, (data, signal) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
                
            data = data.to(device)
            signal = signal.to(device)
            
            # Forward pass
            noise_level = 0.2
            alpha_t = torch.tensor(1.0).to(device)
            
            output, dynamic_alpha = continuous_model(data, signal, noise_level, alpha_t)
            
            # Calculate metrics
            mse = nn.MSELoss()(output, data).item()
            
            total_mse += mse * data.size(0)
            total_samples += data.size(0)
            
            # Store for correlation analysis
            latent = continuous_model.encoder(data)
            latent_mean = latent.mean(dim=(1, 2, 3)).cpu().numpy()
            all_signals.extend(signal.cpu().numpy())
            all_latent_means.extend(latent_mean)
            
            print(f"Batch {i+1} - MSE: {mse:.4f}, Dynamic Alpha: {dynamic_alpha.mean().item():.4f}")
    
    # Calculate average metrics
    avg_mse = total_mse / total_samples
    
    print(f"\nEvaluation Results for Continuous CGCD:")
    print(f"Average MSE: {avg_mse:.4f}")
    
    # Log metrics
    writer.add_scalar("Eval/MSE", avg_mse, 0)
    
    # Correlation analysis
    correlation = np.corrcoef(all_signals, all_latent_means)[0, 1]
    print(f"Signal-Latent Correlation: {correlation:.4f}")
    
    # Visualize correlation
    plt.figure(figsize=(8, 6))
    plt.scatter(all_signals, all_latent_means, alpha=0.5)
    plt.xlabel("Continuous Conditioning Signal")
    plt.ylabel("Average Latent Feature")
    plt.title(f"Latent Feature Alignment with Continuous Condition (r={correlation:.4f})")
    plt.savefig(f"{log_dir}/continuous_correlation.png")
    plt.close()
    
    print(f"Correlation visualization saved to {log_dir}/continuous_correlation.png")
    
    # Conclusion
    print("\nCONCLUSION:")
    if correlation > 0.5:
        print("Strong correlation between conditioning signal and latent features")
        print("The continuous CGCD model successfully aligns latent space with continuous signals")
    elif correlation > 0.2:
        print("Moderate correlation between conditioning signal and latent features")
        print("The continuous CGCD model shows some alignment with continuous signals")
    else:
        print("Weak correlation between conditioning signal and latent features")
        print("The continuous CGCD model struggles to align with continuous signals")
    
    writer.close()
    
    continuous_metrics = {
        'mse': avg_mse,
        'correlation': correlation
    }
    
    print(f"{'='*50}\n")
    return continuous_metrics

def evaluate_continuous_latent_alignment(model, dataloader, device, log_dir):
    """
    Analyze the alignment between continuous signals and latent features.
    
    Args:
        model: Trained continuous CGCD model
        dataloader: DataLoader for continuous data
        device: Device to run analysis on
        log_dir: Directory to save analysis plots
    """
    model.eval()
    all_signals = []
    all_latent_means = []
    
    with torch.no_grad():
        for data, signal in dataloader:
            data = data.to(device)
            signal = signal.to(device)
            
            # Get latent representation
            latent = model.encoder(data)
            latent_mean = latent.mean(dim=(1, 2, 3)).cpu().numpy()
            
            all_signals.extend(signal.cpu().numpy())
            all_latent_means.extend(latent_mean)
    
    # Create plot directory
    os.makedirs(f"{log_dir}/analysis", exist_ok=True)
    
    # Plot the relationship
    plt.figure(figsize=(8, 6))
    plt.scatter(all_signals, all_latent_means, alpha=0.5)
    plt.xlabel("Continuous Conditioning Signal")
    plt.ylabel("Average Latent Feature")
    plt.title("Latent Feature Alignment with Continuous Condition")
    plt.savefig(f"{log_dir}/analysis/latent_alignment.png")
    plt.close()
    
    print(f"Latent alignment analysis completed and saved to {log_dir}/analysis/latent_alignment.png")

def main(test_mode=False):
    """
    Main evaluation function.
    
    Args:
        test_mode: Whether to run in test mode with minimal computations
    """
    from preprocess import get_cifar10_dataloader, get_continuous_dataloader
    
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Get device
    device = get_device()
    
    # Set batch size and max batches for evaluation
    batch_size = 32
    max_batches = 5 if test_mode else None
    
    # Create directories
    log_dir = 'logs'
    model_dir = 'models'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EVALUATING CONDITIONALLY-GUIDED CRITICAL DIFFUSION (CGCD)")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    trainloader, testloader = get_cifar10_dataloader(batch_size=batch_size)
    continuous_loader = get_continuous_dataloader(batch_size=batch_size, n_samples=500)
    
    # Initialize models
    print("\nInitializing models...")
    base_model = BaseDiffusionModel().to(device)
    cgcd_model = CGCDDiffusionModel().to(device)
    
    variant_full = DiffusionModelVariant(adaptive_noise=True, soft_assimilation=True).to(device)
    variant_fixed = DiffusionModelVariant(adaptive_noise=False, soft_assimilation=True).to(device)
    variant_hard = DiffusionModelVariant(adaptive_noise=True, soft_assimilation=False).to(device)
    
    continuous_model = ContinuousCGCDModel().to(device)
    
    # Load trained models if available
    try:
        base_model = load_model(base_model, model_dir, "Base_epoch1.pt")
        cgcd_model = load_model(cgcd_model, model_dir, "CGCD_epoch1.pt")
        variant_full = load_model(variant_full, model_dir, "Variant_Full_epoch1.pt")
        variant_fixed = load_model(variant_fixed, model_dir, "Variant_Fixed_epoch1.pt")
        variant_hard = load_model(variant_hard, model_dir, "Variant_Hard_epoch1.pt")
        continuous_model = load_model(continuous_model, model_dir, "Continuous_epoch1.pt")
    except:
        print("Warning: Could not load trained models. Using untrained models for evaluation.")
    
    # Run evaluations
    print("\nRunning evaluations...")
    
    # Experiment 1: Performance Comparison
    exp1_results = evaluate_experiment1(
        base_model, cgcd_model, testloader,
        max_batches=max_batches, device=device, log_dir=log_dir
    )
    
    # Experiment 2: Ablation Study
    variants = [variant_full, variant_fixed, variant_hard]
    variant_names = ['Variant_Full', 'Variant_Fixed', 'Variant_Hard']
    
    exp2_results = evaluate_experiment2(
        variants, variant_names, testloader,
        max_batches=max_batches, device=device, log_dir=log_dir
    )
    
    # Experiment 3: Continuous Features
    exp3_results = evaluate_experiment3(
        continuous_model, continuous_loader,
        max_batches=max_batches, device=device, log_dir=log_dir
    )
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL EVALUATION SUMMARY")
    print("="*70)
    
    print("\nExperiment 1 (Performance Comparison):")
    print(f"Base Model MSE: {exp1_results['base_metrics']['mse']:.4f}")
    print(f"CGCD Model MSE: {exp1_results['cgcd_metrics']['mse']:.4f}")
    print(f"Improvement: {exp1_results['mse_improvement']:.4f} ({exp1_results['mse_improvement']/exp1_results['base_metrics']['mse']*100:.2f}%)")
    
    print("\nExperiment 2 (Ablation Study):")
    for name, metrics in exp2_results.items():
        print(f"{name}: MSE = {metrics['mse']:.4f}")
    
    print("\nExperiment 3 (Continuous Features):")
    print(f"MSE: {exp3_results['mse']:.4f}")
    print(f"Signal-Latent Correlation: {exp3_results['correlation']:.4f}")
    
    print("\nCONCLUSION:")
    if exp1_results['mse_improvement'] > 0:
        print("- CGCD outperforms the Base model in reconstruction quality")
    else:
        print("- Base model outperforms CGCD in reconstruction quality")
    
    best_variant = min(exp2_results.items(), key=lambda x: x[1]['mse'])
    print(f"- Best ablation variant: {best_variant[0]}")
    
    if exp3_results['correlation'] > 0.3:
        print("- Continuous CGCD successfully aligns latent space with continuous signals")
    else:
        print("- Continuous CGCD shows limited alignment with continuous signals")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main(test_mode=True)
