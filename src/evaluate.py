"""
Evaluation module for Score-Aligned Step Distillation experiments.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import time
from src.utils.diffusion import compute_kl_loss, compute_score_loss, get_device
from src.utils.visualization import visualize_samples, plot_schedules
from src.train import DiffusionModel


def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate a trained diffusion model.
    
    Args:
        model: Trained DiffusionModel
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
    
    Returns:
        dict: Evaluation metrics
    """
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # For evaluation, use a fixed timestep (e.g., middle of the diffusion process)
            t = torch.ones(images.size(0), device=device).long() * (model.num_steps // 2)
            
            # Forward pass
            output = model(images, t)
            
            # Compute loss
            loss = compute_kl_loss(output, images)
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Evaluating batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation complete. Average Loss: {avg_loss:.4f}")
    
    return {
        'avg_loss': avg_loss
    }


def generate_samples(model, num_samples=16, device='cuda'):
    """
    Generate samples from a trained diffusion model.
    
    Args:
        model: Trained DiffusionModel
        num_samples: Number of samples to generate
        device: Device to run generation on
    
    Returns:
        torch.Tensor: Generated samples
    """
    model.to(device)
    model.eval()
    
    # Determine image size and channels based on model
    if hasattr(model, 'image_size'):
        image_size = model.image_size
    else:
        image_size = 32  # Default to CIFAR-10 size
    
    # Start with random noise
    samples = torch.randn(num_samples, 3, image_size, image_size, device=device)
    
    print("Generating samples...")
    with torch.no_grad():
        # Reverse diffusion process
        for t in range(model.num_steps - 1, -1, -1):
            print(f"Diffusion step {t}/{model.num_steps}")
            
            # Get current timestep
            timesteps = torch.ones(num_samples, device=device).long() * t
            
            # Apply model
            noise_pred = model(samples, timesteps)
            
            # Update samples
            alpha = 1.0 - model.schedule[t]
            samples = (samples - alpha * noise_pred) / (1.0 - alpha)
            
            # Add noise for all but the last step
            if t > 0:
                noise = torch.randn_like(samples) * model.schedule[t-1]
                samples = samples + noise
    
    # Normalize to [0, 1] for visualization
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    
    return samples


def compare_schedules(models_dict, save_path=None):
    """
    Compare schedules from different models.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
        save_path: Path to save the schedule comparison plot
    """
    schedules = []
    labels = []
    
    for name, model in models_dict.items():
        schedules.append(model.schedule.detach().cpu().numpy())
        labels.append(name)
    
    # Plot schedules
    plot_schedules(schedules, labels, title="Comparison of Diffusion Schedules", save_path=save_path)


def evaluate_experiment1_results(results, save_dir='./logs'):
    """
    Evaluate results from Experiment 1: Ablation Study on the Dual-Loss Objective.
    
    Args:
        results: Dictionary of results from experiment1
        save_dir: Directory to save evaluation results
    """
    print("\n=== Evaluating Experiment 1 Results ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract loss histories and final schedules
    lambda_values = []
    loss_histories = []
    final_schedules = []
    
    for key, result in results.items():
        # Extract lambda value from key (format: 'lambda_X.Y')
        lambda_value = float(key.split('_')[1])
        lambda_values.append(lambda_value)
        
        loss_histories.append(result['loss_history'])
        final_schedules.append(result['final_schedule'])
    
    # Sort by lambda value
    sorted_indices = np.argsort(lambda_values)
    lambda_values = [lambda_values[i] for i in sorted_indices]
    loss_histories = [loss_histories[i] for i in sorted_indices]
    final_schedules = [final_schedules[i] for i in sorted_indices]
    
    # Compare final loss values
    final_losses = [history[-1] for history in loss_histories]
    
    print("\nFinal Loss Values:")
    for i, lam in enumerate(lambda_values):
        print(f"λ = {lam}: {final_losses[i]:.4f}")
    
    # Find best lambda value
    best_idx = np.argmin(final_losses)
    best_lambda = lambda_values[best_idx]
    
    print(f"\nBest λ value: {best_lambda} (Loss: {final_losses[best_idx]:.4f})")
    
    # Compare schedules
    print("\nFinal Schedule Comparison:")
    for i, lam in enumerate(lambda_values):
        print(f"λ = {lam}: {final_schedules[i]}")
    
    # Save results to file
    results_path = os.path.join(save_dir, 'experiment1_evaluation.txt')
    with open(results_path, 'w') as f:
        f.write("=== Experiment 1: Ablation Study on the Dual-Loss Objective ===\n\n")
        
        f.write("Final Loss Values:\n")
        for i, lam in enumerate(lambda_values):
            f.write(f"λ = {lam}: {final_losses[i]:.4f}\n")
        
        f.write(f"\nBest λ value: {best_lambda} (Loss: {final_losses[best_idx]:.4f})\n")
        
        f.write("\nFinal Schedule Comparison:\n")
        for i, lam in enumerate(lambda_values):
            f.write(f"λ = {lam}: {final_schedules[i]}\n")
    
    print(f"Evaluation results saved to {results_path}")
    
    return {
        'lambda_values': lambda_values,
        'final_losses': final_losses,
        'best_lambda': best_lambda,
        'final_schedules': final_schedules
    }


def evaluate_experiment2_results(results, save_dir='./logs'):
    """
    Evaluate results from Experiment 2: Learnable Schedule vs. Fixed Schedule.
    
    Args:
        results: Dictionary of results from experiment2
        save_dir: Directory to save evaluation results
    """
    print("\n=== Evaluating Experiment 2 Results ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract loss histories and final schedules
    configs = list(results.keys())
    loss_histories = [results[config]['loss_history'] for config in configs]
    final_schedules = [results[config]['final_schedule'] for config in configs]
    
    # Compare final loss values
    final_losses = [history[-1] for history in loss_histories]
    
    print("\nFinal Loss Values:")
    for i, config in enumerate(configs):
        print(f"{config}: {final_losses[i]:.4f}")
    
    # Find best configuration
    best_idx = np.argmin(final_losses)
    best_config = configs[best_idx]
    
    print(f"\nBest configuration: {best_config} (Loss: {final_losses[best_idx]:.4f})")
    
    # Compare schedules
    print("\nFinal Schedule Comparison:")
    for i, config in enumerate(configs):
        print(f"{config}: {final_schedules[i]}")
    
    # Calculate improvement percentage if learnable is better than fixed
    if 'learnable' in configs and 'fixed' in configs:
        learnable_idx = configs.index('learnable')
        fixed_idx = configs.index('fixed')
        
        learnable_loss = final_losses[learnable_idx]
        fixed_loss = final_losses[fixed_idx]
        
        if learnable_loss < fixed_loss:
            improvement = (fixed_loss - learnable_loss) / fixed_loss * 100
            print(f"\nLearnable schedule improves over fixed schedule by {improvement:.2f}%")
        else:
            improvement = (learnable_loss - fixed_loss) / learnable_loss * 100
            print(f"\nFixed schedule improves over learnable schedule by {improvement:.2f}%")
    
    # Save results to file
    results_path = os.path.join(save_dir, 'experiment2_evaluation.txt')
    with open(results_path, 'w') as f:
        f.write("=== Experiment 2: Learnable Schedule vs. Fixed Schedule ===\n\n")
        
        f.write("Final Loss Values:\n")
        for i, config in enumerate(configs):
            f.write(f"{config}: {final_losses[i]:.4f}\n")
        
        f.write(f"\nBest configuration: {best_config} (Loss: {final_losses[best_idx]:.4f})\n")
        
        f.write("\nFinal Schedule Comparison:\n")
        for i, config in enumerate(configs):
            f.write(f"{config}: {final_schedules[i]}\n")
        
        if 'learnable' in configs and 'fixed' in configs:
            if learnable_loss < fixed_loss:
                f.write(f"\nLearnable schedule improves over fixed schedule by {improvement:.2f}%\n")
            else:
                f.write(f"\nFixed schedule improves over learnable schedule by {improvement:.2f}%\n")
    
    print(f"Evaluation results saved to {results_path}")
    
    return {
        'configs': configs,
        'final_losses': final_losses,
        'best_config': best_config,
        'final_schedules': final_schedules
    }


def evaluate_experiment3_results(results, save_dir='./logs'):
    """
    Evaluate results from Experiment 3: Step Efficiency and Robustness Across Datasets.
    
    Args:
        results: Dictionary of results from experiment3
        save_dir: Directory to save evaluation results
    """
    print("\n=== Evaluating Experiment 3 Results ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract dataset names and step configurations
    datasets = set()
    step_configs = set()
    
    for key in results.keys():
        # Key format: "{dataset_name}_steps{step_count}"
        parts = key.split('_steps')
        dataset = parts[0]
        steps = int(parts[1])
        
        datasets.add(dataset)
        step_configs.add(steps)
    
    datasets = sorted(list(datasets))
    step_configs = sorted(list(step_configs))
    
    # Compare results across datasets and step configurations
    print("\nFinal Loss Values:")
    
    # Create a matrix of results
    loss_matrix = np.zeros((len(datasets), len(step_configs)))
    
    for i, dataset in enumerate(datasets):
        for j, steps in enumerate(step_configs):
            key = f"{dataset}_steps{steps}"
            if key in results:
                loss_history = results[key]['loss_history']
                final_loss = loss_history[-1]
                loss_matrix[i, j] = final_loss
                print(f"{dataset}, steps={steps}: {final_loss:.4f}")
    
    # Find best configuration for each dataset
    print("\nBest Step Configuration per Dataset:")
    for i, dataset in enumerate(datasets):
        best_step_idx = np.argmin(loss_matrix[i, :])
        best_steps = step_configs[best_step_idx]
        best_loss = loss_matrix[i, best_step_idx]
        print(f"{dataset}: steps={best_steps} (Loss: {best_loss:.4f})")
    
    # Find most robust step configuration across datasets
    if len(datasets) > 1:
        avg_losses = np.mean(loss_matrix, axis=0)
        best_avg_idx = np.argmin(avg_losses)
        best_avg_steps = step_configs[best_avg_idx]
        print(f"\nMost robust step configuration across datasets: steps={best_avg_steps} "
              f"(Avg Loss: {avg_losses[best_avg_idx]:.4f})")
    
    # Save results to file
    results_path = os.path.join(save_dir, 'experiment3_evaluation.txt')
    with open(results_path, 'w') as f:
        f.write("=== Experiment 3: Step Efficiency and Robustness Across Datasets ===\n\n")
        
        f.write("Final Loss Values:\n")
        for i, dataset in enumerate(datasets):
            for j, steps in enumerate(step_configs):
                key = f"{dataset}_steps{steps}"
                if key in results:
                    f.write(f"{dataset}, steps={steps}: {loss_matrix[i, j]:.4f}\n")
        
        f.write("\nBest Step Configuration per Dataset:\n")
        for i, dataset in enumerate(datasets):
            best_step_idx = np.argmin(loss_matrix[i, :])
            best_steps = step_configs[best_step_idx]
            best_loss = loss_matrix[i, best_step_idx]
            f.write(f"{dataset}: steps={best_steps} (Loss: {best_loss:.4f})\n")
        
        if len(datasets) > 1:
            avg_losses = np.mean(loss_matrix, axis=0)
            best_avg_idx = np.argmin(avg_losses)
            best_avg_steps = step_configs[best_avg_idx]
            f.write(f"\nMost robust step configuration across datasets: steps={best_avg_steps} "
                  f"(Avg Loss: {avg_losses[best_avg_idx]:.4f})\n")
    
    print(f"Evaluation results saved to {results_path}")
    
    return {
        'datasets': datasets,
        'step_configs': step_configs,
        'loss_matrix': loss_matrix,
    }


def load_model_checkpoint(checkpoint_path, model_class=DiffusionModel, device='cuda'):
    """
    Load a model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_class: Model class to instantiate
        device: Device to load the model on
    
    Returns:
        model: Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model instance
    model = model_class()
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)
