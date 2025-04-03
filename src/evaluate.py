"""
Evaluation module for the Latent-Integrated Fingerprint Diffusion (LIFD) method.

This module provides functionality for:
1. Evaluating the LIFD model's performance
2. Conducting experiments on fingerprint robustness
3. Analyzing the adaptive balancing mechanism
4. Visualizing latent fingerprint injection
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from preprocess import set_seed, simulate_attacks
from train import LIFDModel, FingerprintExtractionNet, CustomCrossAttention, total_variation


def evaluate_fingerprint_extraction(extraction_net, images, true_fingerprints, threshold=0.5):
    """
    Evaluate fingerprint extraction accuracy.
    
    Args:
        extraction_net (nn.Module): Fingerprint extraction network
        images (torch.Tensor): Generated images
        true_fingerprints (torch.Tensor): Ground truth fingerprints
        threshold (float): Threshold for binary classification
        
    Returns:
        float: Extraction accuracy
    """
    extraction_net.eval()
    
    with torch.no_grad():
        extracted_fingerprints = extraction_net(images)
        
        pred_fingerprints = (torch.sigmoid(extracted_fingerprints) > threshold).float()
        
        accuracy = (pred_fingerprints == true_fingerprints).float().mean().item()
    
    return accuracy

def evaluate_image_quality(images, reference_images=None):
    """
    Evaluate image quality using MSE and PSNR.
    
    Args:
        images (torch.Tensor): Generated images
        reference_images (torch.Tensor): Reference images for comparison
        
    Returns:
        tuple: (mse, psnr) - Mean squared error and peak signal-to-noise ratio
    """
    if reference_images is None:
        return 0.0, float('inf')
    
    with torch.no_grad():
        mse = F.mse_loss(images, reference_images).item()
        
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    return mse, psnr

def evaluate_model(lifd_model, extraction_net, test_data, fingerprints, config=None):
    """
    Evaluate the LIFD model on test data.
    
    Args:
        lifd_model (nn.Module): LIFD model
        extraction_net (nn.Module): Fingerprint extraction network
        test_data (list): List of test data batches
        fingerprints (torch.Tensor): Tensor of fingerprints
        config (dict): Configuration parameters
        
    Returns:
        dict: Evaluation metrics
    """
    if config is None:
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_users': 10,
            'mode': 'C',
            'adaptive_alpha': 0.5
        }
    
    device = torch.device(config['device'])
    
    lifd_model.eval()
    extraction_net.eval()
    
    lifd_model.set_mode(config['mode'])
    lifd_model.set_adaptive_balance(config['adaptive_alpha'])
    
    metrics = {
        'clean_extraction_acc': 0.0,
        'attacked_blur_jpeg_acc': 0.0,
        'attacked_noise_acc': 0.0,
        'mse': 0.0,
        'psnr': 0.0
    }
    
    num_batches = len(test_data)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data):
            batch = batch.to(device)
            
            batch_size = batch.size(0)
            user_indices = torch.randint(0, config['num_users'], (batch_size,))
            batch_fingerprints = fingerprints[user_indices].to(device)
            
            generated_images = lifd_model(batch, batch_fingerprints)
            
            mse, psnr = evaluate_image_quality(generated_images, batch)
            
            clean_acc = evaluate_fingerprint_extraction(extraction_net, generated_images, batch_fingerprints)
            
            attacked_image, attacked_noisy = simulate_attacks(generated_images[0])
            
            attacked_acc = evaluate_fingerprint_extraction(
                extraction_net,
                attacked_image.unsqueeze(0).to(device),
                batch_fingerprints[0].unsqueeze(0)
            )
            
            noisy_acc = evaluate_fingerprint_extraction(
                extraction_net,
                attacked_noisy.unsqueeze(0).to(device),
                batch_fingerprints[0].unsqueeze(0)
            )
            
            metrics['clean_extraction_acc'] += clean_acc
            metrics['attacked_blur_jpeg_acc'] += attacked_acc
            metrics['attacked_noise_acc'] += noisy_acc
            metrics['mse'] += mse
            metrics['psnr'] += psnr
            
            print(f"Batch {batch_idx+1}/{num_batches} - "
                  f"Clean Acc: {clean_acc:.4f}, Blur/JPEG Acc: {attacked_acc:.4f}, "
                  f"Noise Acc: {noisy_acc:.4f}, MSE: {mse:.4f}, PSNR: {psnr:.2f} dB")
    
    for key in metrics:
        metrics[key] /= num_batches
    
    print("\nEvaluation Summary:")
    print(f"Clean Extraction Accuracy: {metrics['clean_extraction_acc']:.4f}")
    print(f"Attacked (Blur+JPEG) Accuracy: {metrics['attacked_blur_jpeg_acc']:.4f}")
    print(f"Attacked (Noise) Accuracy: {metrics['attacked_noise_acc']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    
    return metrics


def dual_channel_fingerprint_experiment(lifd_model, extraction_net, test_data, fingerprints, config=None):
    """
    Experiment 1: Compare three modes of fingerprint injection under simulated attacks.
    Modes: 'A' = Parameter-Only, 'B' = Latent-Only, 'C' = Dual-Channel LIFD.
    
    Args:
        lifd_model (nn.Module): LIFD model
        extraction_net (nn.Module): Fingerprint extraction network
        test_data (list): List of test data batches
        fingerprints (torch.Tensor): Tensor of fingerprints
        config (dict): Configuration parameters
        
    Returns:
        dict: Experiment results
    """
    print("\n--- Starting Dual-Channel Fingerprint Robustness Experiment ---")
    
    if config is None:
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_users': 10,
            'adaptive_alpha': 0.5,
            'log_dir': 'logs'
        }
    
    device = torch.device(config['device'])
    
    lifd_model.eval()
    extraction_net.eval()
    
    os.makedirs(config['log_dir'], exist_ok=True)
    
    modes = ['A', 'B', 'C']
    results = {mode: {"clean": [], "attacked_blur_jpeg": [], "attacked_noise": []} for mode in modes}
    
    for mode in modes:
        print(f"\nProcessing Mode {mode}")
        lifd_model.set_mode(mode)
        
        for i in range(min(5, len(test_data))):
            batch = test_data[i].to(device)
            
            batch_size = batch.size(0)
            user_indices = torch.randint(0, config['num_users'], (batch_size,))
            batch_fingerprints = fingerprints[user_indices].to(device)
            
            with torch.no_grad():
                generated_images = lifd_model(batch, batch_fingerprints)
            
            clean_acc = evaluate_fingerprint_extraction(extraction_net, generated_images, batch_fingerprints)
            
            attacked_image, attacked_noisy = simulate_attacks(generated_images[0])
            
            attacked_acc = evaluate_fingerprint_extraction(
                extraction_net,
                attacked_image.unsqueeze(0).to(device),
                batch_fingerprints[0].unsqueeze(0)
            )
            
            noisy_acc = evaluate_fingerprint_extraction(
                extraction_net,
                attacked_noisy.unsqueeze(0).to(device),
                batch_fingerprints[0].unsqueeze(0)
            )
            
            results[mode]["clean"].append(clean_acc)
            results[mode]["attacked_blur_jpeg"].append(attacked_acc)
            results[mode]["attacked_noise"].append(noisy_acc)
            
            print(f"Iteration {i+1}: Clean Acc: {clean_acc:.3f}, "
                  f"Blurred/JPEG Acc: {attacked_acc:.3f}, Noisy Acc: {noisy_acc:.3f}")
    
    avg_results = {}
    for mode in modes:
        avg_results[mode] = {
            "clean": np.mean(results[mode]["clean"]),
            "attacked_blur_jpeg": np.mean(results[mode]["attacked_blur_jpeg"]),
            "attacked_noise": np.mean(results[mode]["attacked_noise"]),
        }
        print(f"\nMode {mode} averages: {avg_results[mode]}")
    
    plt.figure(figsize=(10, 6))
    index = np.arange(len(modes))
    bar_width = 0.25
    
    clean_bar = [avg_results[m]["clean"] for m in modes]
    blur_bar = [avg_results[m]["attacked_blur_jpeg"] for m in modes]
    noise_bar = [avg_results[m]["attacked_noise"] for m in modes]
    
    plt.bar(index, clean_bar, bar_width, label='Clean')
    plt.bar(index + bar_width, blur_bar, bar_width, label='Blur+JPEG')
    plt.bar(index + 2*bar_width, noise_bar, bar_width, label='Noise')
    
    plt.xlabel('Modes')
    plt.ylabel('Fingerprint Extraction Accuracy')
    plt.title('Dual-Channel Fingerprint Robustness Analysis')
    plt.xticks(index + bar_width, modes)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['log_dir'], 'fingerprint_robustness.pdf'), format='pdf', dpi=300)
    print(f"Dual-Channel Fingerprint Robustness plot saved as {os.path.join(config['log_dir'], 'fingerprint_robustness.pdf')}")
    
    return avg_results


def ablation_study_experiment(lifd_model, extraction_net, test_data, fingerprints, config=None):
    """
    Experiment 2: Evaluate the effect of the adaptive balancing parameter (α) on image quality
    and on fingerprint extraction accuracy.
    
    Args:
        lifd_model (nn.Module): LIFD model
        extraction_net (nn.Module): Fingerprint extraction network
        test_data (list): List of test data batches
        fingerprints (torch.Tensor): Tensor of fingerprints
        config (dict): Configuration parameters
        
    Returns:
        dict: Experiment results
    """
    print("\n--- Starting Ablation Study on Adaptive Balancing ---")
    
    if config is None:
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_users': 10,
            'log_dir': 'logs'
        }
    
    device = torch.device(config['device'])
    
    lifd_model.eval()
    extraction_net.eval()
    
    os.makedirs(config['log_dir'], exist_ok=True)
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    mse_scores = []
    psnr_scores = []
    fingerprint_accuracies = []
    
    lifd_model.set_mode('C')
    
    batch = test_data[0].to(device)
    
    batch_size = batch.size(0)
    user_indices = torch.randint(0, config['num_users'], (batch_size,))
    batch_fingerprints = fingerprints[user_indices].to(device)
    
    for alpha in alphas:
        print(f"\nEvaluating for alpha = {alpha}")
        lifd_model.set_adaptive_balance(alpha)
        
        with torch.no_grad():
            generated_images = lifd_model(batch, batch_fingerprints)
        
        mse, psnr = evaluate_image_quality(generated_images, batch)
        
        acc = evaluate_fingerprint_extraction(extraction_net, generated_images, batch_fingerprints)
        
        mse_scores.append(mse)
        psnr_scores.append(psnr)
        fingerprint_accuracies.append(acc)
        
        print(f"Alpha={alpha}: MSE={mse:.4f}, PSNR={psnr:.2f} dB, Fingerprint Acc={acc:.3f}")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(alphas, mse_scores, marker='o', label='MSE (lower is better)')
    plt.xlabel('Alpha (Adaptive Balancing Parameter)')
    plt.ylabel('MSE')
    plt.title('Image Quality (MSE) vs Adaptive Balancing')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(alphas, psnr_scores, marker='s', label='PSNR (higher is better)')
    plt.xlabel('Alpha (Adaptive Balancing Parameter)')
    plt.ylabel('PSNR (dB)')
    plt.title('Image Quality (PSNR) vs Adaptive Balancing')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['log_dir'], 'image_quality_adaptive_balance.pdf'), format='pdf', dpi=300)
    print(f"Ablation Study image quality plot saved as {os.path.join(config['log_dir'], 'image_quality_adaptive_balance.pdf')}")
    
    plt.figure()
    plt.plot(alphas, fingerprint_accuracies, marker='o', label='Fingerprint Extraction Accuracy')
    plt.xlabel('Alpha (Adaptive Balancing Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Fingerprint Extraction Accuracy vs Adaptive Balancing')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config['log_dir'], 'fingerprint_accuracy_adaptive_balance.pdf'), format='pdf', dpi=300)
    print(f"Ablation Study fingerprint accuracy plot saved as {os.path.join(config['log_dir'], 'fingerprint_accuracy_adaptive_balance.pdf')}")
    
    results = {
        'alphas': alphas,
        'mse_scores': mse_scores,
        'psnr_scores': psnr_scores,
        'fingerprint_accuracies': fingerprint_accuracies
    }
    
    return results


def latent_fingerprint_injection_analysis(lifd_model, test_data, fingerprints, config=None):
    """
    Experiment 3: Hook into the cross-attention block to capture latent attention maps,
    visualize the spatial distribution, and calculate total variation.
    
    Args:
        lifd_model (nn.Module): LIFD model
        test_data (list): List of test data batches
        fingerprints (torch.Tensor): Tensor of fingerprints
        config (dict): Configuration parameters
        
    Returns:
        dict: Experiment results
    """
    print("\n--- Starting Latent Fingerprint Injection Analysis ---")
    
    if config is None:
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_users': 10,
            'log_dir': 'logs'
        }
    
    device = torch.device(config['device'])
    
    lifd_model.eval()
    
    os.makedirs(config['log_dir'], exist_ok=True)
    
    lifd_model.set_mode('B')
    
    lifd_model.cross_attention.attention_map = None
    
    batch = test_data[0].to(device)
    
    batch_size = batch.size(0)
    user_indices = torch.randint(0, config['num_users'], (batch_size,))
    batch_fingerprints = fingerprints[user_indices].to(device)
    
    with torch.no_grad():
        generated_images = lifd_model(batch, batch_fingerprints)
    
    attn_map = lifd_model.cross_attention.attention_map
    if attn_map is None:
        print("No attention map found. Check the cross-attention module.")
        return None
    
    print("Attention map shape:", attn_map.shape)
    
    attn_np = attn_map[0].squeeze().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_np, cmap='hot', interpolation='nearest')
    plt.title('Attention Map Heatmap')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(config['log_dir'], 'attention_map_heatmap.pdf'), format='pdf', dpi=300)
    print(f"Latent Fingerprint Injection heatmap saved as {os.path.join(config['log_dir'], 'attention_map_heatmap.pdf')}")
    
    tv_val = total_variation(attn_map)
    print("Total Variation of the attention map:", tv_val.item())
    
    plt.figure()
    plt.bar(["Attention Map"], [tv_val.item()])
    plt.ylabel("Total Variation")
    plt.title("Sharpness of Attention Map")
    plt.tight_layout()
    plt.savefig(os.path.join(config['log_dir'], 'attention_tv.pdf'), format='pdf', dpi=300)
    print(f"Attention map total variation plot saved as {os.path.join(config['log_dir'], 'attention_tv.pdf')}")
    
    results = {
        'attention_map': attn_map.cpu(),
        'total_variation': tv_val.item()
    }
    
    return results

def evaluate_lifd(trained_models, test_data, config=None):
    """
    Main evaluation function that runs all experiments.
    
    Args:
        trained_models (dict): Dictionary containing trained models
        test_data (list): List of test data batches
        config (dict): Configuration parameters
        
    Returns:
        dict: Evaluation results
    """
    if config is None:
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_users': 10,
            'adaptive_alpha': 0.5,
            'log_dir': 'logs'
        }
    
    lifd_model = trained_models['lifd_model']
    extraction_net = trained_models['extraction_net']
    fingerprints = trained_models.get('fingerprints', None)
    
    if fingerprints is None:
        from preprocess import generate_fingerprints
        fingerprints = generate_fingerprints(num_users=config['num_users'], 
                                           fingerprint_dim=config.get('fingerprint_dim', 128))
    
    os.makedirs(config['log_dir'], exist_ok=True)
    
    print("\n=== Running General Evaluation ===")
    general_metrics = evaluate_model(lifd_model, extraction_net, test_data, fingerprints, config)
    
    print("\n=== Running Experiment 1: Dual-Channel Fingerprint Robustness ===")
    robustness_results = dual_channel_fingerprint_experiment(lifd_model, extraction_net, test_data, fingerprints, config)
    
    print("\n=== Running Experiment 2: Ablation Study on Adaptive Balancing ===")
    ablation_results = ablation_study_experiment(lifd_model, extraction_net, test_data, fingerprints, config)
    
    print("\n=== Running Experiment 3: Latent Fingerprint Injection Analysis ===")
    injection_results = latent_fingerprint_injection_analysis(lifd_model, test_data, fingerprints, config)
    
    evaluation_results = {
        'general_metrics': general_metrics,
        'robustness_results': robustness_results,
        'ablation_results': ablation_results,
        'injection_results': injection_results
    }
    
    print("\nAll evaluations completed successfully")
    return evaluation_results

if __name__ == "__main__":
    from preprocess import preprocess_data
    from train import train_model
    
    config = {
        'seed': 42,
        'batch_size': 16,
        'image_size': 64,
        'use_dummy': True,
        'data_dir': None,
        'fingerprint_dim': 128,
        'num_users': 10,
        'num_epochs': 1,  # Small number for testing
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'adaptive_alpha': 0.5,
        'mode': 'C',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'models',
        'log_dir': 'logs'
    }
    
    preprocessed_data = preprocess_data(config)
    
    trained_models = train_model(preprocessed_data, config)
    
    evaluation_results = evaluate_lifd(
        trained_models=trained_models,
        test_data=preprocessed_data['val_data'],  # Use validation data as test data for testing
        config=config
    )
    
    print("Evaluation test completed successfully")
