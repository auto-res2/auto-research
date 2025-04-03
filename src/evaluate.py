"""
Evaluation module for the Purify-Tweedie++ experiment.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchattacks
import torch.nn as nn
from tqdm import tqdm

def generate_adversaries(model, data_loader, attack_name='PGD', eps=8/255, alpha=2/255, steps=10, device="cuda"):
    """
    Generate adversarial examples using a chosen attack.
    
    Args:
        model (torch.nn.Module): Target model
        data_loader (DataLoader): Data loader
        attack_name (str): Name of the attack ('FGSM', 'PGD', or 'CW')
        eps (float): Maximum perturbation
        alpha (float): Step size for PGD
        steps (int): Number of steps for PGD
        device (str): Device to use
        
    Returns:
        tuple: (adversarial_images, original_labels, original_images)
    """
    model.eval()
    
    if attack_name.upper() == 'FGSM':
        attack = torchattacks.FGSM(model, eps=eps)
    elif attack_name.upper() == 'PGD':
        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    elif attack_name.upper() == 'CW':
        attack = torchattacks.CW(model)
    else:
        raise ValueError(f"Attack {attack_name} is not supported")
    
    adv_images_list = []
    labels_list = []
    orig_images_list = []
    
    for data, target in tqdm(data_loader, desc=f"Generating {attack_name} adversaries"):
        data, target = data.to(device), target.to(device)
        adv = attack(data, target)
        
        adv_images_list.append(adv.cpu())
        labels_list.append(target.cpu())
        orig_images_list.append(data.cpu())
        
    return (torch.cat(adv_images_list), 
            torch.cat(labels_list), 
            torch.cat(orig_images_list))

def evaluate_model(model, data_loader, device="cuda"):
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (DataLoader): Data loader
        device (str): Device to use
        
    Returns:
        float: Accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    return correct / total

def experiment_ablation(model, test_loader, clean_images, adv_images, labels, device="cuda"):
    """
    Run ablation study of the novel components.
    
    Args:
        model (torch.nn.Module): Target model
        test_loader (DataLoader): Test data loader
        clean_images (torch.Tensor): Original clean images
        adv_images (torch.Tensor): Adversarial images
        labels (torch.Tensor): True labels
        device (str): Device to use
        
    Returns:
        dict: Results of the ablation study
    """
    from train import PurifyTweediePlusPlus
    
    configs = {
        'full': {},
        'no_double_tweedie': {'disable_double_tweedie': True},
        'no_consistency': {'disable_consistency_loss': True},
        'no_adaptive_cov': {'disable_adaptive_cov': True}
    }
    
    results = {}
    for cfg_name, params in configs.items():
        pipeline = PurifyTweediePlusPlus(model=model, device=device, **params)
        result = pipeline.purify(adv_images.to(device))
        if isinstance(result, tuple) and len(result) == 3:
            purified_images, uncertainties, _ = result
        else:
            purified_images, uncertainties = result
        
        with torch.no_grad():
            model.eval()
            preds = model(purified_images)
            acc = (preds.argmax(dim=1).cpu() == labels).float().mean().item()
        
        mse_loss = nn.MSELoss()
        mse = mse_loss(purified_images.cpu(), clean_images[:purified_images.shape[0]]).item()
        
        results[cfg_name] = {
            'accuracy': acc, 
            'reconstruction_error': mse, 
            'uncertainty': uncertainties.mean().item()
        }
        
        print(f"Config: {cfg_name:>20s} | Accuracy: {acc:.4f} | MSE: {mse:.6f} | "
              f"Uncertainty: {uncertainties.mean().item():.4f}")
            
    return results

def plot_ablation_results(results, output_dir="logs"):
    """
    Plot ablation study results and save as PDF.
    
    Args:
        results (dict): Results from experiment_ablation
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    configs = list(results.keys())
    accuracies = [results[c]['accuracy'] for c in configs]
    errors = [results[c]['reconstruction_error'] for c in configs]
    
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=configs, y=accuracies)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Ablation Study: Classification Accuracy", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_accuracy.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=configs, y=errors)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Ablation Study: Reconstruction Error", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_reconstruction_error.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def experiment_robustness(model, test_loader, device="cuda"):
    """
    Run robustness experiment against diverse adversarial threat models.
    
    Args:
        model (torch.nn.Module): Target model
        test_loader (DataLoader): Test data loader
        device (str): Device to use
        
    Returns:
        dict: Results of the robustness experiment
    """
    from train import PurifyTweediePlusPlus
    
    print("\n=== Experiment 2: Robustness Against Diverse Adversarial Threat Models ===")
    
    attack_list = {
        'FGSM': {'attack_name': 'FGSM', 'eps': 8/255},
        'PGD': {'attack_name': 'PGD', 'eps': 8/255, 'alpha': 2/255, 'steps': 10},
        'CW': {'attack_name': 'CW'}  # CW attack doesn't use epsilon
    }
    
    performance_metrics = {}
    
    for attack_name, attack_params in attack_list.items():
        print(f"\nEvaluating attack method: {attack_name}")
        adv_images, labels, clean_images = generate_adversaries(
            model, test_loader, device=device, **attack_params
        )
        
        performance_metrics[attack_name] = {}
        for method in ['Base', 'Purify-Tweedie++']:
            if method == 'Base':
                pipeline = PurifyTweediePlusPlus(
                    model=model, device=device,
                    disable_double_tweedie=True,
                    disable_consistency_loss=True,
                    disable_adaptive_cov=True
                )
            else:
                pipeline = PurifyTweediePlusPlus(model=model, device=device)
                
            result = pipeline.purify(adv_images.to(device))
            if isinstance(result, tuple) and len(result) == 3:
                purified_images, uncertainties, _ = result
            else:
                purified_images, uncertainties = result
            
            with torch.no_grad():
                model.eval()
                preds = model(purified_images)
                acc = (preds.argmax(dim=1).cpu() == labels).float().mean().item()
            
            performance_metrics[attack_name][method] = acc
            print(f"Attack: {attack_name:4s} | Method: {method:20s} | Accuracy: {acc:.4f}")
            
    return performance_metrics

def plot_robustness_results(results, output_dir="logs"):
    """
    Plot robustness experiment results and save as PDF.
    
    Args:
        results (dict): Results from experiment_robustness
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    methods = ['Base', 'Purify-Tweedie++']
    attack_names = list(results.keys())
    acc_values = {m: [results[a][m] for a in attack_names] for m in methods}
    
    sns.set_style("whitegrid")
    
    x = np.arange(len(attack_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, acc_values['Base'], width, label='Base', alpha=0.8)
    plt.bar(x + width/2, acc_values['Purify-Tweedie++'], width, label='Purify-Tweedie++', alpha=0.8)
    
    plt.xlabel("Attack Method", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Robustness Against Diverse Adversarial Threats", fontsize=14)
    plt.xticks(x, attack_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/robustness_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def experiment_efficiency(model, test_loader, device="cuda"):
    """
    Run efficiency and uncertainty-guided recovery analysis.
    
    Args:
        model (torch.nn.Module): Target model
        test_loader (DataLoader): Test data loader
        device (str): Device to use
        
    Returns:
        dict: Results of the efficiency experiment
    """
    from train import PurifyTweediePlusPlus
    
    print("\n=== Experiment 3: Efficiency and Uncertainty-guided Recovery Analysis ===")
    
    adv_images, labels, clean_images = generate_adversaries(
        model, test_loader, attack_name='PGD', 
        eps=8/255, alpha=2/255, steps=10,
        device=device
    )
    
    profiling_results = {}
    
    for method in ['Base', 'Purify-Tweedie++']:
        if method == 'Base':
            pipeline = PurifyTweediePlusPlus(
                model=model, device=device,
                disable_double_tweedie=True,
                disable_consistency_loss=True,
                disable_adaptive_cov=True
            )
        else:
            pipeline = PurifyTweediePlusPlus(model=model, device=device)
            
        start_time = time.time()
        purified_images, uncertainties, step_logs = pipeline.purify(adv_images.to(device), log_steps=True)
        total_time = time.time() - start_time
        
        steps = [log['time_elapsed'] for log in step_logs]
        step_uncertainties = [log['uncertainty'].mean().item() for log in step_logs]
        
        mse_loss = nn.MSELoss()
        step_errors = [
            mse_loss(log['partial_output'].cpu(), clean_images[:log['partial_output'].size(0)]).item() 
            for log in step_logs
        ]
        
        profiling_results[method] = {
            'total_time': total_time,
            'steps': steps,
            'uncertainties': step_uncertainties,
            'errors': step_errors
        }
        
        print(f"Method: {method:20s} | Total Purification Time: {total_time:.4f} sec")
        
    return profiling_results

def plot_efficiency_results(results, output_dir="logs"):
    """
    Plot efficiency experiment results and save as PDF.
    
    Args:
        results (dict): Results from experiment_efficiency
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for method, res in results.items():
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        ax[0].plot(res['steps'], res['uncertainties'], marker='o', linewidth=2)
        ax[0].set_title(f"{method} - Uncertainty Evolution", fontsize=14)
        ax[0].set_xlabel("Time (sec)", fontsize=12)
        ax[0].set_ylabel("Mean Uncertainty", fontsize=12)
        ax[0].grid(alpha=0.3)
        
        ax[1].plot(res['steps'], res['errors'], marker='o', linewidth=2)
        ax[1].set_title(f"{method} - Reconstruction Error Evolution", fontsize=14)
        ax[1].set_xlabel("Time (sec)", fontsize=12)
        ax[1].set_ylabel("Reconstruction MSE", fontsize=12)
        ax[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/efficiency_{method.replace(' ', '_').lower()}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
