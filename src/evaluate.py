"""
Model evaluation for PurifyCov++ experiments.
"""

import os
import torch
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from utils.models import SimpleClassifier, CovarianceNet
from utils.attacks import generate_fgsm_examples
from utils.purification import purify_diffusion
from utils.visualization import save_boxplot, save_line_plot
from utils.data import get_test_loader
from train import load_models

def experiment1(device, test_loader, classifier, covariance_net, timesteps=20):
    """
    Experiment 1: Comparison under Varied Adversarial Attacks.
    
    Args:
        device: Device to run on (cuda or cpu)
        test_loader: DataLoader for test data
        classifier: The classifier model
        covariance_net: The covariance prediction network
        timesteps: Number of diffusion timesteps
        
    Returns:
        results: Dictionary of experiment results
    """
    print("=== Experiment 1: Adversarial Attack Comparison ===")
    classifier.eval()
    covariance_net.eval()
    
    robust_acc_cov = 0
    robust_acc_fixed = 0
    total = 0
    batch_times_cov = []
    batch_times_fixed = []

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        adv_images = generate_fgsm_examples(classifier, images, labels, epsilon=0.03)

        start_cov = time.time()
        purified_cov = purify_diffusion(adv_images, timesteps=timesteps, method='cov', covariance_net=covariance_net)
        time_cov = time.time() - start_cov

        start_fixed = time.time()
        purified_fixed = purify_diffusion(adv_images, timesteps=timesteps, method='fixed')
        time_fixed = time.time() - start_fixed

        preds_cov = classifier(purified_cov).argmax(dim=1)
        preds_fixed = classifier(purified_fixed).argmax(dim=1)
        robust_acc_cov += (preds_cov == labels).sum().item()
        robust_acc_fixed += (preds_fixed == labels).sum().item()
        total += labels.size(0)
        batch_times_cov.append(time_cov)
        batch_times_fixed.append(time_fixed)
        print(f"Batch {batch_idx+1}: Time Cov: {time_cov:.4f}s, Time Fixed: {time_fixed:.4f}s")

    acc_cov_percent = 100.0 * robust_acc_cov / total
    acc_fixed_percent = 100.0 * robust_acc_fixed / total
    
    print(f"PurifyCov++ Robust Accuracy: {acc_cov_percent:.2f}%")
    print(f"Purify++ Robust Accuracy: {acc_fixed_percent:.2f}%")

    save_boxplot(
        [batch_times_cov, batch_times_fixed], 
        ['Cov++', 'Fixed'],
        'Purification Runtime Comparison',
        'Purification Time (s) per Batch',
        "logs/purification_runtime_comparison.pdf"
    )
    
    return {'acc_cov': acc_cov_percent, 'acc_fixed': acc_fixed_percent}

def experiment2(device, test_loader, classifier, covariance_net, timesteps=20, time_points=None):
    """
    Experiment 2: Ablation Study on the Covariance Prediction Module.
    
    Args:
        device: Device to run on (cuda or cpu)
        test_loader: DataLoader for test data
        classifier: The classifier model
        covariance_net: The covariance prediction network
        timesteps: Number of diffusion timesteps
        time_points: List of timesteps at which to record PSNR
        
    Returns:
        results: Dictionary of experiment results
    """
    print("=== Experiment 2: Ablation Study on Covariance Prediction Module ===")
    
    if time_points is None:
        time_points = [0, 5, 10, 15, 20]

    classifier.eval()
    covariance_net.eval()
    
    psnr_trajectory_adaptive = []
    psnr_trajectory_fixed = []
    
    images, _ = next(iter(test_loader))
    images = images.to(device)
    
    adv_images = generate_fgsm_examples(classifier, images, torch.zeros(images.size(0), dtype=torch.long, device=device), epsilon=0.03)
    
    purified_adaptive = adv_images.clone()
    purified_fixed = adv_images.clone()
    
    for t in range(timesteps+1):
        if t > 0:
            sigma_a = covariance_net(purified_adaptive, t)
            noise_a = torch.randn_like(purified_adaptive) * sigma_a
            purified_adaptive = purified_adaptive - 0.1 * purified_adaptive + noise_a
            
            sigma_f = torch.ones_like(purified_fixed) * 0.1
            noise_f = torch.randn_like(purified_fixed) * sigma_f
            purified_fixed = purified_fixed - 0.1 * purified_fixed + noise_f
            
        if t in time_points:
            psnr_adaptive = []
            psnr_fixed = []
            purified_adaptive_cpu = purified_adaptive.detach().cpu().numpy()
            purified_fixed_cpu = purified_fixed.detach().cpu().numpy()
            clean_cpu = images.detach().cpu().numpy()
            for i in range(len(clean_cpu)):
                psnr_a = compute_psnr(np.transpose(clean_cpu[i], (1, 2, 0)),
                                     np.transpose(purified_adaptive_cpu[i], (1, 2, 0)))
                psnr_f = compute_psnr(np.transpose(clean_cpu[i], (1, 2, 0)),
                                     np.transpose(purified_fixed_cpu[i], (1, 2, 0)))
                psnr_adaptive.append(psnr_a)
                psnr_fixed.append(psnr_f)
            avg_psnr_a = np.mean(psnr_adaptive)
            avg_psnr_f = np.mean(psnr_fixed)
            psnr_trajectory_adaptive.append(avg_psnr_a)
            psnr_trajectory_fixed.append(avg_psnr_f)
            print(f"Step {t}: Adaptive PSNR = {avg_psnr_a:.2f}, Fixed PSNR = {avg_psnr_f:.2f}")

    save_line_plot(
        time_points,
        [psnr_trajectory_adaptive, psnr_trajectory_fixed],
        ['PurifyCov++ (Adaptive)', 'Purify++ (Fixed)'],
        'Ablation Study: PSNR Trajectory',
        'Diffusion Steps',
        'Average PSNR',
        "logs/psnr_trajectory_ablation.pdf"
    )

    return {
        'time_points': time_points,
        'psnr_adaptive': psnr_trajectory_adaptive,
        'psnr_fixed': psnr_trajectory_fixed
    }

def experiment3(device, test_loader, classifier, covariance_net, diffusion_steps_list=None):
    """
    Experiment 3: Efficiency and Convergence Analysis.
    
    Args:
        device: Device to run on (cuda or cpu)
        test_loader: DataLoader for test data
        classifier: The classifier model
        covariance_net: The covariance prediction network
        diffusion_steps_list: List of diffusion steps to test
        
    Returns:
        results: Dictionary of experiment results
    """
    print("=== Experiment 3: Efficiency and Convergence Analysis ===")
    
    if diffusion_steps_list is None:
        diffusion_steps_list = [10, 20, 30]
        
    classifier.eval()
    covariance_net.eval()
    
    results = {
        'steps': [],
        'accuracy_cov': [],
        'accuracy_fixed': [],
        'psnr_cov': [],
        'psnr_fixed': [],
        'time_cov': [],
        'time_fixed': []
    }
    
    for steps in diffusion_steps_list:
        acc_cov = 0
        acc_fixed = 0
        total_images = 0
        total_time_cov = 0.0
        total_time_fixed = 0.0
        psnr_cov_vals = []
        psnr_fixed_vals = []
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            total_images += labels.size(0)
            
            adv_images = generate_fgsm_examples(classifier, images, labels, epsilon=0.03)
            
            t0 = time.time()
            purified_cov = purify_diffusion(adv_images, steps, method='cov', covariance_net=covariance_net)
            elapsed_cov = time.time() - t0
            total_time_cov += elapsed_cov
            
            t0 = time.time()
            purified_fixed = purify_diffusion(adv_images, steps, method='fixed')
            elapsed_fixed = time.time() - t0
            total_time_fixed += elapsed_fixed
            
            preds_cov = classifier(purified_cov).argmax(dim=1)
            preds_fixed = classifier(purified_fixed).argmax(dim=1)
            acc_cov += (preds_cov == labels).sum().item()
            acc_fixed += (preds_fixed == labels).sum().item()
            
            purified_cov_cpu = purified_cov.detach().cpu().numpy()
            purified_fixed_cpu = purified_fixed.detach().cpu().numpy()
            clean_cpu = images.detach().cpu().numpy()
            for i in range(len(clean_cpu)):
                psnr_cov_vals.append(compute_psnr(np.transpose(clean_cpu[i], (1,2,0)),
                                                 np.transpose(purified_cov_cpu[i], (1,2,0))))
                psnr_fixed_vals.append(compute_psnr(np.transpose(clean_cpu[i], (1,2,0)),
                                                   np.transpose(purified_fixed_cpu[i], (1,2,0))))
                
        acc_cov_percent = 100.0 * acc_cov / total_images
        acc_fixed_percent = 100.0 * acc_fixed / total_images
        avg_psnr_cov = np.mean(psnr_cov_vals)
        avg_psnr_fixed = np.mean(psnr_fixed_vals)
        avg_time_cov = total_time_cov / len(test_loader)
        avg_time_fixed = total_time_fixed / len(test_loader)
        
        results['steps'].append(steps)
        results['accuracy_cov'].append(acc_cov_percent)
        results['accuracy_fixed'].append(acc_fixed_percent)
        results['psnr_cov'].append(avg_psnr_cov)
        results['psnr_fixed'].append(avg_psnr_fixed)
        results['time_cov'].append(avg_time_cov)
        results['time_fixed'].append(avg_time_fixed)
        
        print(f"Diffusion Steps: {steps} | Adaptive Accuracy: {acc_cov_percent:.2f}%, "
              f"Fixed Accuracy: {acc_fixed_percent:.2f}%")
        print(f"           | Adaptive PSNR: {avg_psnr_cov:.2f}, Fixed PSNR: {avg_psnr_fixed:.2f}")
        print(f"           | Adaptive Time per Batch: {avg_time_cov:.4f}s, Fixed: {avg_time_fixed:.4f}s")
    
    save_line_plot(
        results['steps'],
        [results['accuracy_cov'], results['accuracy_fixed']],
        ['PurifyCov++ Accuracy', 'Purify++ Accuracy'],
        'Accuracy vs. Diffusion Steps',
        'Diffusion Steps',
        'Classification Accuracy (%)',
        "logs/accuracy_diffusion_steps.pdf"
    )

    save_line_plot(
        results['steps'],
        [results['psnr_cov'], results['psnr_fixed']],
        ['PurifyCov++ PSNR', 'Purify++ PSNR'],
        'PSNR vs. Diffusion Steps',
        'Diffusion Steps',
        'Average PSNR',
        "logs/psnr_diffusion_steps.pdf"
    )
    
    return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_loader = get_test_loader(batch_size=32, quick_test=True)
    
    classifier, covariance_net = load_models(device)
    
    experiment1(device, test_loader, classifier, covariance_net)
    experiment2(device, test_loader, classifier, covariance_net)
    experiment3(device, test_loader, classifier, covariance_net)
