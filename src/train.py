"""
Training module for HBFN experiments.
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import sys

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.models import (
    BaselineEncoder, HyperbolicEncoder, Decoder, FlowNetwork,
    reconstruction_loss, hyperbolic_regulated_loss
)
from utils.evaluation import evaluate_topology, plot_latents, plot_loss_curves

def train_topology_experiment(tree_graph, dataset, data_loader, config):
    """
    Train models for the Topological Structure Preservation experiment.
    
    Args:
        tree_graph: NetworkX graph with tree structure
        dataset: PyTorch dataset with node features
        data_loader: PyTorch dataloader
        config: Configuration dictionary
    
    Returns:
        baseline_model: Trained baseline BFN model
        hyperbolic_model: Trained HBFN model
        baseline_loss_history: Training loss history for baseline model
        hyper_loss_history: Training loss history for hyperbolic model
    """
    print("\n***** Starting Experiment 1: Topological Structure Preservation *****")
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    input_dim = config["feature_dim"]
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    
    baseline_encoder = BaselineEncoder(input_dim, hidden_dim, latent_dim)
    decoder_base = Decoder(latent_dim, hidden_dim, input_dim)
    model_baseline = FlowNetwork(baseline_encoder, decoder_base, device)
    model_baseline.to(device)
    
    hyper_encoder = HyperbolicEncoder(input_dim, hidden_dim, latent_dim)
    decoder_hyper = Decoder(latent_dim, hidden_dim, input_dim)
    model_hyper = FlowNetwork(hyper_encoder, decoder_hyper, device)
    model_hyper.to(device)
    
    optimizer_base = optim.Adam(model_baseline.parameters(), lr=config["learning_rate"])
    optimizer_hyper = optim.Adam(model_hyper.parameters(), lr=config["learning_rate"])
    
    num_epochs = config["num_epochs"]
    baseline_loss_history = []
    hyper_loss_history = []
    baseline_corr_history = []
    hyper_corr_history = []
    
    for epoch in range(num_epochs):
        model_baseline.train()
        model_hyper.train()
        total_loss_base = 0.0
        total_loss_hyper = 0.0
        
        for batch in data_loader:
            batch = batch.to(device)
            
            optimizer_base.zero_grad()
            recon_base, mu_base, _ = model_baseline(batch)
            loss_base = reconstruction_loss(recon_base, batch)
            loss_base.backward()
            optimizer_base.step()
            total_loss_base += loss_base.item()
            
            optimizer_hyper.zero_grad()
            recon_hyper, mu_hyper, _ = model_hyper(batch)
            loss_hyper = reconstruction_loss(recon_hyper, batch)
            loss_hyper.backward()
            optimizer_hyper.step()
            total_loss_hyper += loss_hyper.item()
        
        avg_loss_base = total_loss_base / len(data_loader)
        avg_loss_hyper = total_loss_hyper / len(data_loader)
        baseline_loss_history.append(avg_loss_base)
        hyper_loss_history.append(avg_loss_hyper)
        
        corr_base = evaluate_topology(model_baseline, dataset, tree_graph, device)
        corr_hyper = evaluate_topology(model_hyper, dataset, tree_graph, device)
        baseline_corr_history.append(corr_base)
        hyper_corr_history.append(corr_hyper)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Baseline Loss={avg_loss_base:.4f}, HBFN Loss={avg_loss_hyper:.4f}")
        print(f"Topology Correlation: Baseline={corr_base:.4f}, HBFN={corr_hyper:.4f}")
    
    plot_latents(model_baseline, dataset, "Baseline BFN Latent Embeddings", 
               "latent_embeddings_baseline_pair2.pdf", device)
    plot_latents(model_hyper, dataset, "HBFN Latent Embeddings", 
               "latent_embeddings_hyperbolic_pair2.pdf", device)
    
    plot_loss_curves(
        baseline_loss_history, 
        hyper_loss_history,
        ["Baseline BFN", "HBFN"],
        "Reconstruction Loss Comparison",
        "training_loss_topological_structure_pair1.pdf"
    )
    
    print("Experiment 1 completed. Loss and latent embedding plots saved as PDF files.")
    
    return model_baseline, model_hyper, baseline_loss_history, hyper_loss_history
