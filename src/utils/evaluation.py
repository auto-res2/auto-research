"""
Evaluation utilities for HBFN experiments.
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

def evaluate_topology(model, dataset, tree_graph, device='cpu'):
    """
    Measure the correlation between pairwise latent distances and
    the true shortest path distances in the tree graph.
    """
    model.eval()
    latent_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x = dataset[i].unsqueeze(0).to(device)
            _, latent, _ = model(x)
            latent_list.append(latent.squeeze(0).cpu().numpy())
    
    latents = np.stack(latent_list)  # shape: (num_nodes, latent_dim)
    
    diff = latents[:, np.newaxis, :] - latents[np.newaxis, :, :]
    latent_dists = np.linalg.norm(diff, axis=2)
    
    num_nodes = latents.shape[0]
    gt_dists = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        lengths = nx.single_source_shortest_path_length(tree_graph, i)
        for j in range(num_nodes):
            gt_dists[i, j] = lengths[j]
    
    iu = np.triu_indices(num_nodes, k=1)
    corr = np.corrcoef(latent_dists[iu], gt_dists[iu])[0, 1]
    
    return corr

def compute_fid(generated_images, iteration):
    """
    Dummy FID computation. In a real implementation, this would use a 
    proper FID score calculator.
    """
    fid = max(100.0 / (iteration + 1), 5.0)
    return fid

def plot_latents(model, dataset, title, filename, device='cpu'):
    """
    Plot latent embeddings (2D scatter) for visualization.
    """
    model.eval()
    latent_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x = dataset[i].unsqueeze(0).to(device)
            _, latent, _ = model(x)
            latent_list.append(latent.squeeze(0).cpu().numpy())
    
    latents = np.stack(latent_list)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(latents[:, 0], latents[:, 1], c='blue', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.tight_layout()
    
    os.makedirs('logs', exist_ok=True)
    
    plt.savefig(os.path.join('logs', filename), format='pdf', dpi=300)
    plt.close()

def plot_loss_curves(losses_a, losses_b, labels, title, filename):
    """
    Plot training loss curves for comparison.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(losses_a, label=labels[0])
    plt.plot(losses_b, label=labels[1])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('logs', exist_ok=True)
    
    plt.savefig(os.path.join('logs', filename), format='pdf', dpi=300)
    plt.close()
