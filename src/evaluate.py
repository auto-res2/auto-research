"""
Evaluation module for the NTEC-G experiment.
Includes functions for analysis and visualization of results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
import os

os.makedirs('logs', exist_ok=True)

def plot_convergence_curves(norm_diffs_base, probe_norms, filename="training_loss_comparison.pdf"):
    """
    Plot convergence curves comparing Base Method and NTEC-G.
    
    Args:
        norm_diffs_base: Norm differences from Base Method
        probe_norms: Norm differences from NTEC-G probes
        filename: Output filename for the PDF plot
    """
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=list(range(len(norm_diffs_base))), y=norm_diffs_base, label="Base Method")
    sns.scatterplot(x=list(range(len(probe_norms))), y=probe_norms, color="red", label="NTEC-G (Probes)")
    plt.xlabel("Iteration")
    plt.ylabel("Norm Difference")
    plt.title("Convergence Behavior Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('logs', filename), dpi=300)
    plt.close()
    print(f"Convergence plot saved as {os.path.join('logs', filename)}.")

def plot_samples_grid(samples, title="Samples", filename="samples_grid.pdf", nrow=8):
    """
    Plot a grid of sample images and save as PDF.
    
    Args:
        samples: Tensor of samples (N, 3, H, W)
        title: Plot title
        filename: Output filename for the PDF plot
        nrow: Number of samples per row
    """
    num_samples = samples.shape[0]
    ncol = int(np.ceil(num_samples / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*1.5, nrow*1.5))
    axes = np.array(axes).reshape(-1)
    
    for idx, ax in enumerate(axes):
        if idx < num_samples:
            img = samples[idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")
            
    plt.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join('logs', filename), dpi=300)
    plt.close()
    print(f"Sample grid for '{title}' saved as {os.path.join('logs', filename)}.")

def visualize_latent_space(latents, title="Latent Space", filename="latent_space.pdf"):
    """
    Project latent representations with t-SNE and save as PDF.
    
    Args:
        latents: Latent representations
        title: Plot title
        filename: Output filename for the PDF plot
    """
    latents_np = latents.detach().cpu().numpy()
    
    latents_np = np.nan_to_num(latents_np, nan=0.0, posinf=1.0, neginf=-1.0)
    
    latents_np = np.clip(latents_np, -1e6, 1e6)
    
    n_samples = latents_np.shape[0]
    perplexity = min(30, n_samples - 1)  # Default is 30, but ensure it's less than n_samples
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)  
    z_proj = tsne.fit_transform(latents_np)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(z_proj[:, 0], z_proj[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join('logs', filename), dpi=300)
    plt.close()
    print(f"t-SNE latent space plot saved as {os.path.join('logs', filename)}.")

def generate_samples(guidance_func, model, num_samples=32, device='cpu'):
    """
    Generate samples by running a simplified diffusion process.
    
    Args:
        guidance_func: Guidance function (base_method or ntec_g)
        model: Guidance model
        num_samples: Number of samples to generate
        device: Device to run the generation on
        
    Returns:
        torch.Tensor: Generated samples
    """
    samples = []
    for idx in range(num_samples):
        state = torch.randn(1, 128).to(device)
        for t in range(10):
            state = guidance_func(state, model)
        state_flat = state.view(-1)
        repeated = state_flat.repeat((3072 // state_flat.shape[0] + 1))[:3072]
        image = repeated.view(3, 32, 32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        samples.append(image.cpu())
        if idx < 3:  # print for first few samples
            print(f"[Sample Generation] Sample {idx} generated.")
    
    samples_tensor = torch.stack(samples)
    return samples_tensor
