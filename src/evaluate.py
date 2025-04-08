import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.hyperbolic import hyperbolic_distance

def fixed_point_inference(model, data, init_noise_level, max_iterations=20, tol=1e-4):
    """
    Perform fixed-point iteration inference for denoising.
    
    Args:
        model: Diffusion model instance
        data: Graph data
        init_noise_level: Initial noise level
        max_iterations: Maximum number of iterations (default: 20)
        tol: Convergence tolerance (default: 1e-4)
        
    Returns:
        Denoised latent representation, number of iterations taken, and iteration history
    """
    latent = model.encoder(data.x, data.edge_index)
    noisy_latent = latent + torch.randn_like(latent) * init_noise_level
    current_latent = noisy_latent.clone()
    iteration_history = []
    
    for i in range(max_iterations):
        updated_latent = model.decoder(current_latent)
        dist = F.mse_loss(current_latent, updated_latent)
        iteration_history.append(dist.item())
        
        if dist < tol:
            break
            
        current_latent = updated_latent
    
    return current_latent, i + 1, iteration_history

def save_experiment_results(experiment_name, plot_data, xlabel, ylabel, title, filename):
    """
    Save experiment results as a high-quality PDF plot.
    
    Args:
        experiment_name: Name of the experiment
        plot_data: Data to plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    if isinstance(plot_data, dict) and 'type' in plot_data and plot_data['type'] == 'heatmap':
        sns.heatmap(
            plot_data['data'], 
            annot=True, 
            xticklabels=plot_data.get('xticklabels', None),
            yticklabels=plot_data.get('yticklabels', None),
            cmap="viridis"
        )
    elif isinstance(plot_data, dict) and 'type' in plot_data and plot_data['type'] == 'barplot':
        sns.barplot(x=plot_data['x'], y=plot_data['y'])
    else:
        plt.plot(np.array(plot_data), marker="o")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{experiment_name}: {title}")
    plt.tight_layout()
    plt.savefig(f"logs/{filename}.pdf")
    plt.close()
    
    print(f"Experiment plot saved as logs/{filename}.pdf")
