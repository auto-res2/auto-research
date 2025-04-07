"""
Evaluation functions for Cov-Purify++ experiments.
"""
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

from src.utils.diffusion import purify_fixed, cov_purify_dynamic, adaptive_reverse_diffusion, fixed_reverse_diffusion
from src.utils.attacks import generate_adversarial_examples
from src.utils.metrics import evaluate_quality, compute_robust_accuracy
from src.utils.plotting import plot_comparison_bar, plot_line, plot_heatmap

def experiment1_comparative_robustness(model, dataloader, device):
    """
    Experiment 1: Comparative Robustness Evaluation.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to run on
        
    Returns:
        dict: Results of the experiment
    """
    print("Starting Experiment 1: Comparative Robustness Evaluation")
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        adv_images = generate_adversarial_examples(model, images, labels, attack_type='pgd')
        
        start_time = time.time()
        purified_fixed_result = purify_fixed(adv_images)
        time_fixed = time.time() - start_time
        
        start_time = time.time()
        purified_dynamic = cov_purify_dynamic(adv_images)
        time_dynamic = time.time() - start_time
        
        acc_fixed = compute_robust_accuracy(model, purified_fixed_result, labels)
        acc_dynamic = compute_robust_accuracy(model, purified_dynamic, labels)
        
        l2_error_fixed = torch.norm((purified_fixed_result - images).view(images.size(0), -1), dim=1).mean().item()
        l2_error_dynamic = torch.norm((purified_dynamic - images).view(images.size(0), -1), dim=1).mean().item()
        
        results = {
            'acc_fixed': acc_fixed,
            'acc_dynamic': acc_dynamic,
            'l2_error_fixed': l2_error_fixed,
            'l2_error_dynamic': l2_error_dynamic,
            'time_fixed': time_fixed,
            'time_dynamic': time_dynamic,
        }
        
        print("Comparative Robustness Evaluation Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        labels_x = ['Purify++ (Fixed)', 'Cov-Purify++ (Dynamic)']
        acc_means = [acc_fixed, acc_dynamic]
        plot_comparison_bar(
            labels_x, acc_means, 
            ylabel="Robust Accuracy",
            title="Robust Accuracy Comparison",
            filename="accuracy_comparison_pair1"
        )
        
        runtime_means = [time_fixed*1000, time_dynamic*1000]  # in milliseconds
        plot_comparison_bar(
            labels_x, runtime_means, 
            ylabel="Runtime (ms)",
            title="Purification Runtime Comparison",
            filename="inference_latency_comparison_pair1"
        )
        
        break  # Process only one batch for quick evaluation
    
    return results

def experiment2_adaptive_solver(device):
    """
    Experiment 2: Adaptive Numerical Solver & Step-Adaptation Analysis.
    
    Args:
        device (torch.device): Device to run on
        
    Returns:
        dict: Results of the experiment
    """
    print("Starting Experiment 2: Adaptive Numerical Solver & Step-Adaptation Analysis")
    
    x_dummy = torch.randn(16, 3, 32, 32).to(device)
    
    start_time = time.time()
    purified_adaptive, step_sizes = adaptive_reverse_diffusion(x_dummy)
    adaptive_time = time.time() - start_time
    
    start_time = time.time()
    purified_fixed = fixed_reverse_diffusion(x_dummy)
    fixed_time = time.time() - start_time
    
    error_adaptive = torch.norm(purified_adaptive).item()
    error_fixed = torch.norm(purified_fixed).item()
    
    results = {
        'step_sizes': step_sizes,
        'total_iterations_adaptive': len(step_sizes),
        'total_iterations_fixed': 50,
        'runtime_adaptive': adaptive_time,
        'runtime_fixed': fixed_time,
        'error_adaptive': error_adaptive,
        'error_fixed': error_fixed,
    }
    
    print("Adaptive Reverse Diffusion Results:")
    print(f"  Step sizes selected (per iteration): {step_sizes}")
    print(f"  Total iterations: {len(step_sizes)}")
    print(f"  Runtime: {adaptive_time:.4f} seconds, Reconstruction error: {error_adaptive:.4f}")
    
    print("Fixed Reverse Diffusion Results:")
    print(f"  Total iterations: 50, Runtime: {fixed_time:.4f} seconds, Reconstruction error: {error_fixed:.4f}")
    
    plot_line(
        range(1, len(step_sizes)+1), step_sizes, 
        xlabel="Iteration",
        ylabel="Step Size",
        title="Adaptive Step Size Evolution in Reverse Diffusion",
        filename="step_adaptation_pair1"
    )
    
    return results

def experiment3_hyperparameter_sensitivity(device):
    """
    Experiment 3: Hyperparameter Sensitivity & Covariance Estimation Analysis.
    
    Args:
        device (torch.device): Device to run on
        
    Returns:
        dict: Results of the experiment
    """
    print("Starting Experiment 3: Hyperparameter Sensitivity & Covariance Estimation Analysis")
    
    noise_levels = [0.01, 0.03, 0.05]
    fixed_lambdas = [0.2, 0.5, 0.8]
    
    results_hyper = []
    
    for noise_level in noise_levels:
        for fixed_lambda in fixed_lambdas:
            clean_batch = torch.randn(16, 3, 32, 32).to(device)
            adv_batch = clean_batch + noise_level * torch.randn_like(clean_batch)
            
            purified_fixed = purify_fixed(adv_batch, lambda_value=fixed_lambda)
            psnr_fixed, ssim_fixed = evaluate_quality(clean_batch, purified_fixed)
            
            purified_dynamic = cov_purify_dynamic(adv_batch)
            psnr_dynamic, ssim_dynamic = evaluate_quality(clean_batch, purified_dynamic)
            
            result = {
                'noise_level': noise_level,
                'fixed_lambda': fixed_lambda,
                'psnr_fixed': psnr_fixed,
                'ssim_fixed': ssim_fixed,
                'psnr_dynamic': psnr_dynamic,
                'ssim_dynamic': ssim_dynamic,
            }
            
            results_hyper.append(result)
            print(f"Setting: noise_level={noise_level}, fixed_lambda={fixed_lambda}")
            print(f"  PSNR (fixed): {psnr_fixed:.2f}, SSIM (fixed): {ssim_fixed:.4f}")
            print(f"  PSNR (dynamic): {psnr_dynamic:.2f}, SSIM (dynamic): {ssim_dynamic:.4f}")
    
    sample_img = clean_batch[0].unsqueeze(0)  # take one image from the last batch
    local_cov = torch.var(sample_img, dim=[2, 3]).squeeze().cpu().detach().numpy()
    cov_matrix = np.zeros((3, 3))
    for i in range(min(3, len(local_cov))):
        cov_matrix[i, i] = local_cov[i]
    
    plot_heatmap(
        cov_matrix, 
        title="Local Covariance Estimation Heatmap",
        filename="covariance_estimation_heatmap_pair1"
    )
    
    return results_hyper
