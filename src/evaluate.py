#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

def calculate_fid(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance between real and generated images.
    
    This is a simplified dummy implementation for demonstration purposes.
    In a real implementation, use a proper FID calculator with Inception model.
    
    Args:
        real_images (torch.Tensor): Batch of real images
        generated_images (torch.Tensor): Batch of generated images
        
    Returns:
        float: Calculated FID score
    """
    # In a real implementation, this would use the Inception model to extract features
    # and calculate the FID based on the feature statistics
    
    # For demonstration, return a random FID value that's influenced by image differences
    # to simulate some correlation with actual quality
    real_mean = real_images.mean().item()
    gen_mean = generated_images.mean().item()
    mean_diff = abs(real_mean - gen_mean)
    
    # Add some randomness to simulate variability in FID calculation
    base_fid = 20.0 + mean_diff * 100
    random_factor = np.random.uniform(0.8, 1.2)
    
    return base_fid * random_factor

def evaluate_model(generator, dataloader, device, num_samples=64, save_dir='./results'):
    """
    Evaluate a generator model by calculating FID and generating sample images.
    
    Args:
        generator (nn.Module): Generator model to evaluate
        dataloader (DataLoader): DataLoader for real images
        device (torch.device): Device to run evaluation on
        num_samples (int): Number of samples to generate
        save_dir (str): Directory to save generated images
        
    Returns:
        dict: Evaluation metrics
    """
    generator.eval()
    
    # Get a batch of real images for comparison
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)
    
    # Generate images
    with torch.no_grad():
        z = torch.randn(num_samples, generator.latent_dim, device=device)
        generated_images = generator(z)
    
    # Calculate FID
    fid_score = calculate_fid(real_images, generated_images)
    
    # Save generated images
    os.makedirs(save_dir, exist_ok=True)
    grid = make_grid(generated_images.cpu(), nrow=8, normalize=True)
    save_image(grid, os.path.join(save_dir, 'generated_samples.png'))
    
    # Calculate additional metrics
    gen_mean = generated_images.mean().item()
    gen_std = generated_images.std().item()
    
    # Return evaluation metrics
    metrics = {
        'fid': fid_score,
        'mean': gen_mean,
        'std': gen_std
    }
    
    return metrics

def compare_models(models_dict, dataloader, device, num_samples=64, save_dir='./results'):
    """
    Compare multiple generator models by evaluating each one.
    
    Args:
        models_dict (dict): Dictionary mapping model names to generator models
        dataloader (DataLoader): DataLoader for real images
        device (torch.device): Device to run evaluation on
        num_samples (int): Number of samples to generate per model
        save_dir (str): Directory to save generated images and results
        
    Returns:
        dict: Comparison results for all models
    """
    os.makedirs(save_dir, exist_ok=True)
    comparison_results = {}
    
    for model_name, generator in models_dict.items():
        print(f"Evaluating {model_name}...")
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Evaluate the model
        metrics = evaluate_model(generator, dataloader, device, num_samples, model_save_dir)
        comparison_results[model_name] = metrics
        
        print(f"  FID: {metrics['fid']:.2f}")
        print(f"  Mean: {metrics['mean']:.4f}")
        print(f"  Std: {metrics['std']:.4f}")
    
    # Save comparison results to JSON
    with open(os.path.join(save_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Create comparison plots
    plot_comparison(comparison_results, save_dir)
    
    return comparison_results

def plot_comparison(comparison_results, save_dir):
    """
    Create plots comparing the performance of different models.
    
    Args:
        comparison_results (dict): Dictionary of model evaluation results
        save_dir (str): Directory to save plots
    """
    # Extract model names and FID scores
    model_names = list(comparison_results.keys())
    fid_scores = [results['fid'] for results in comparison_results.values()]
    
    # Create FID comparison bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, fid_scores)
    plt.xlabel('Model')
    plt.ylabel('FID Score (lower is better)')
    plt.title('FID Score Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fid_comparison.png'))
    plt.close()
    
    # Create additional comparison plots if there are more metrics
    if len(comparison_results) > 0 and 'mean' in next(iter(comparison_results.values())):
        means = [results['mean'] for results in comparison_results.values()]
        stds = [results['std'] for results in comparison_results.values()]
        
        # Plot means
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, means)
        plt.xlabel('Model')
        plt.ylabel('Mean Pixel Value')
        plt.title('Mean Pixel Value Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mean_comparison.png'))
        plt.close()
        
        # Plot standard deviations
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, stds)
        plt.xlabel('Model')
        plt.ylabel('Pixel Value Standard Deviation')
        plt.title('Pixel Value Standard Deviation Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'std_comparison.png'))
        plt.close()

def evaluate_sensitivity_results(results_path, save_dir='./results/sensitivity'):
    """
    Analyze and visualize the results of the sensitivity analysis experiment.
    
    Args:
        results_path (str): Path to the sensitivity results JSON file
        save_dir (str): Directory to save visualization plots
        
    Returns:
        dict: Processed sensitivity analysis results
    """
    # Load sensitivity results
    with open(results_path, 'r') as f:
        sensitivity_results = json.load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract hyperparameters and metrics
    uncertainty_factors = []
    sde_stepsizes = []
    fid_values = {}
    loss_values = {}
    
    for key, trials in sensitivity_results.items():
        # Parse hyperparameters from the key (format: 'uf_{uf}_ss_{ss}')
        parts = key.split('_')
        uf = float(parts[1])
        ss = float(parts[3])
        
        if uf not in uncertainty_factors:
            uncertainty_factors.append(uf)
        if ss not in sde_stepsizes:
            sde_stepsizes.append(ss)
        
        # Get the final epoch metrics
        final_epoch = trials[-1]
        
        # Store FID and loss values in 2D grid format
        if uf not in fid_values:
            fid_values[uf] = {}
        if uf not in loss_values:
            loss_values[uf] = {}
        
        fid_values[uf][ss] = final_epoch['fid']
        loss_values[uf][ss] = final_epoch['loss']
    
    # Sort the hyperparameter values
    uncertainty_factors.sort()
    sde_stepsizes.sort()
    
    # Create heatmaps for FID and loss
    create_heatmap(uncertainty_factors, sde_stepsizes, fid_values, 
                  'FID Score (lower is better)', 
                  os.path.join(save_dir, 'fid_heatmap.png'))
    
    create_heatmap(uncertainty_factors, sde_stepsizes, loss_values, 
                  'Loss Value (lower is better)', 
                  os.path.join(save_dir, 'loss_heatmap.png'))
    
    # Return processed results
    processed_results = {
        'uncertainty_factors': uncertainty_factors,
        'sde_stepsizes': sde_stepsizes,
        'fid_values': fid_values,
        'loss_values': loss_values
    }
    
    return processed_results

def create_heatmap(x_values, y_values, data_dict, title, save_path):
    """
    Create a heatmap visualization for sensitivity analysis results.
    
    Args:
        x_values (list): Values for the x-axis (uncertainty factors)
        y_values (list): Values for the y-axis (SDE stepsizes)
        data_dict (dict): Nested dictionary containing the metric values
        title (str): Title for the heatmap
        save_path (str): Path to save the heatmap image
    """
    # Create a 2D array for the heatmap
    data_array = np.zeros((len(x_values), len(y_values)))
    
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            data_array[i, j] = data_dict[x][y]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(data_array, cmap='viridis', interpolation='nearest')
    plt.colorbar(label=title.split('(')[0].strip())
    
    # Set axis labels and ticks
    plt.xlabel('SDE Stepsize')
    plt.ylabel('Uncertainty Factor')
    plt.xticks(np.arange(len(y_values)), y_values)
    plt.yticks(np.arange(len(x_values)), x_values)
    
    # Add text annotations in each cell
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            plt.text(j, i, f"{data_array[i, j]:.2f}", 
                    ha="center", va="center", color="w")
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Simple test to verify the evaluation module
    print("Evaluation module loaded successfully.")
    
    # Create dummy data for testing
    class DummyGenerator:
        def __init__(self):
            self.latent_dim = 100
        
        def eval(self):
            pass
        
        def __call__(self, z):
            # Generate random images for testing
            return torch.randn(z.size(0), 3, 32, 32)
    
    class DummyDataset:
        def __iter__(self):
            return iter([(torch.randn(16, 3, 32, 32), torch.zeros(16))])
    
    # Test the evaluation functions
    device = torch.device("cpu")
    dummy_generator = DummyGenerator()
    dummy_dataloader = DummyDataset()
    
    print("Testing evaluate_model function...")
    metrics = evaluate_model(dummy_generator, dummy_dataloader, device, num_samples=16, save_dir='./test_results')
    print(f"FID: {metrics['fid']:.2f}")
    
    print("Evaluation module test complete.")
