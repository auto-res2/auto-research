"""
Evaluation module for HBFN experiments.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.models import (
    HyperbolicEncoder, Decoder, FlowNetwork, HyperbolicSDESolver,
    reconstruction_loss, hyperbolic_regulated_loss
)
from utils.evaluation import compute_fid, plot_loss_curves

def evaluate_sampling_efficiency(config):
    """
    Evaluate sampling efficiency and fidelity for Experiment 2.
    
    Args:
        config: Configuration dictionary
    """
    print("\n***** Starting Experiment 2: Sampling Efficiency and Fidelity *****")
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    
    hyper_encoder = HyperbolicEncoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)
    model = FlowNetwork(hyper_encoder, decoder, device)
    model.to(device)
    
    model.eval()
    
    sde_solver = HyperbolicSDESolver(model, num_iter=max(config["iterations_to_test"]), device=device)
    
    iterations_to_test = config["iterations_to_test"]
    fid_scores = []
    
    for n_iter in iterations_to_test:
        print(f"Testing with {n_iter} iterations...")
        
        sde_solver.num_iter = n_iter
        
        batch_size = config["batch_size"]
        sample_latents = sde_solver.sample(batch_size, latent_dim)
        
        with torch.no_grad():
            generated_images = model.decoder(sample_latents)
        
        fid = compute_fid(generated_images, n_iter)
        fid_scores.append(fid)
        
        print(f"Iterations: {n_iter}, Simulated FID Score: {fid:.4f}")
    
    plt.figure(figsize=(6, 4))
    plt.plot(iterations_to_test, fid_scores, marker='o')
    plt.xlabel("Number of SDE Iterations")
    plt.ylabel("Simulated FID Score")
    plt.title("Sampling Efficiency (FID vs Iterations)")
    plt.tight_layout()
    
    os.makedirs('logs', exist_ok=True)
    
    plt.savefig(os.path.join('logs', 'inference_latency_hyperbolic_pair1.pdf'), 
               format='pdf', dpi=300)
    plt.close()
    
    print("Experiment 2 completed. FID vs iterations plot saved as PDF file.")
    
    return fid_scores

def evaluate_loss_impact(dataset, data_loader, config):
    """
    Evaluate the impact of hyperbolic-regulated loss for Experiment 3.
    
    Args:
        dataset: PyTorch dataset with node features
        data_loader: PyTorch dataloader
        config: Configuration dictionary
    """
    print("\n***** Starting Experiment 3: Combined Loss Impact and Optimization Stability *****")
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    input_dim = config["feature_dim"]
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    
    hyper_encoder = HyperbolicEncoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)
    model = FlowNetwork(hyper_encoder, decoder, device)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    num_epochs = config["num_epochs"]
    loss_history_standard = []
    loss_history_hyper = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss_standard = 0.0
        total_loss_hyper = 0.0
        
        for batch in data_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            recon, latent, _ = model(batch)
            loss_standard = reconstruction_loss(recon, batch)
            loss_standard.backward()
            optimizer.step()
            total_loss_standard += loss_standard.item()
            
            optimizer.zero_grad()
            recon2, latent2, _ = model(batch)
            loss_denoise = reconstruction_loss(recon2, batch)
            
            expected_angles = torch.zeros(latent2.size(0), device=device)
            expected_radii = torch.ones(latent2.size(0), device=device)
            
            loss_hyper_reg = hyperbolic_regulated_loss(
                latent2, expected_angles, expected_radii, model.encoder.manifold)
            
            loss_total = loss_denoise + config["hyperbolic_loss_weight"] * loss_hyper_reg
            loss_total.backward()
            optimizer.step()
            
            total_loss_hyper += loss_total.item()
        
        avg_loss_standard = total_loss_standard / len(data_loader)
        avg_loss_hyper = total_loss_hyper / len(data_loader)
        loss_history_standard.append(avg_loss_standard)
        loss_history_hyper.append(avg_loss_hyper)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Standard Loss={avg_loss_standard:.4f}, "
              f"HBFN + Hyper Loss={avg_loss_hyper:.4f}")
    
    plot_loss_curves(
        loss_history_standard,
        loss_history_hyper,
        ["Standard Denoising Loss", "Denoising + Hyperbolic Loss"],
        "Training Loss Comparison",
        "training_loss_hyperbolic_loss_pair1.pdf"
    )
    
    print("Experiment 3 completed. Training loss curves saved as PDF file.")
    
    return loss_history_standard, loss_history_hyper
