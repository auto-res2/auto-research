"""
Main script for Geometrically Consistent Ambient Diffusion (GCAD) experiment.

This script runs three experiments:
1. Training Efficiency and Denoising Quality Comparison
2. Hyperbolic Latent Space Analysis
3. Robustness to Limited and Linearly Corrupted Data

All plots are saved as PDF files for academic papers.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import lpips

from .preprocess import get_cifar10_data, corrupt_images
from .train import (
    BaseAutoencoder,
    GCADModel,
    GCADModelAblation,
    GCADHyperbolicEncoder,
    train_model,
    extract_latents,
    compute_psnr
)
from .evaluate import evaluate_model, evaluate_lpips

os.makedirs("logs", exist_ok=True)


def experiment1(train_loader, val_loader, device, num_epochs=10, test_mode=False):
    """
    Experiment 1: Compare training convergence and reconstruction quality (MSE, PSNR, SSIM)
    between BaseAutoencoder and GCADModel using CIFAR-10 with Gaussian noise.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run the experiment on
        num_epochs: Number of epochs for training
        test_mode: If True, fewer epochs are run for testing
        
    Returns:
        None (saves plots to logs directory)
    """
    epochs = 1 if test_mode else num_epochs

    print("\n==== Experiment 1: BaseAutoencoder Training (Gaussian Noise) ====")
    base_model = BaseAutoencoder()
    base_model, base_loss = train_model(base_model, train_loader, device, noise_type="gaussian", num_epochs=epochs, record_loss=True)
    base_mse, base_psnr, base_ssim = evaluate_model(base_model, val_loader, device, noise_type="gaussian")
    print(f"Base Model: MSE={base_mse:.4f}, PSNR={base_psnr:.2f}, SSIM={base_ssim:.4f}")

    print("\n==== Experiment 1: GCADModel Training (Gaussian Noise) ====")
    gcad_model = GCADModel()
    gcad_model, gcad_loss = train_model(gcad_model, train_loader, device, noise_type="gaussian", num_epochs=epochs, record_loss=True)
    gcad_mse, gcad_psnr, gcad_ssim = evaluate_model(gcad_model, val_loader, device, noise_type="gaussian")
    print(f"GCAD Model: MSE={gcad_mse:.4f}, PSNR={gcad_psnr:.2f}, SSIM={gcad_ssim:.4f}")

    plt.figure(figsize=(8, 5))
    if base_loss is not None:
        plt.plot(range(1, epochs+1), base_loss, label="BaseAutoencoder")
    if gcad_loss is not None:
        plt.plot(range(1, epochs+1), gcad_loss, label="GCADModel")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch (Gaussian Noise)")
    plt.legend()
    plt.tight_layout()
    pdf_filename = "logs/training_loss_gaussian_pair1.pdf"
    plt.savefig(pdf_filename)
    print(f"Saved training loss plot as {pdf_filename}")
    plt.close()


def experiment2(val_loader, device, test_mode=False):
    """
    Experiment 2: Compare latent representations between the Base method and GCAD with a hyperbolic encoder.
    We use t-SNE to reduce dimensions, plot the embeddings, and compute silhouette scores.
    
    Args:
        val_loader: DataLoader for validation data
        device: Device to run the experiment on
        test_mode: If True, fewer samples are used for testing
        
    Returns:
        None (saves plots to logs directory)
    """
    base_model = BaseAutoencoder().to(device)
    hyperbolic_encoder = GCADHyperbolicEncoder(latent_dim=32).to(device)
    
    print("\n==== Experiment 2: Extracting Latent Features ====")
    base_latents, base_labels = extract_latents(base_model, val_loader, device, noise_type="gaussian", 
                                                  encoder_fn=lambda x: base_model.encoder(x).view(x.size(0), -1))
    gcad_latents, gcad_labels = extract_latents(base_model, val_loader, device, noise_type="gaussian", 
                                                encoder_fn=lambda x: hyperbolic_encoder(x))
    
    print("Computing t-SNE for latent representations...")
    tsne = TSNE(n_components=2, random_state=42)
    base_tsne = tsne.fit_transform(base_latents)
    gcad_tsne = tsne.fit_transform(gcad_latents)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(base_tsne[:,0], base_tsne[:,1], c=base_labels, cmap="tab10", s=5)
    plt.title("Base Method Latent Space (t-SNE)")
    plt.colorbar(sc1)
    
    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(gcad_tsne[:,0], gcad_tsne[:,1], c=gcad_labels, cmap="tab10", s=5)
    plt.title("GCAD Hyperbolic Latent Space (t-SNE)")
    plt.colorbar(sc2)
    plt.tight_layout()
    pdf_filename = "logs/latent_space_tsne_pair1.pdf"
    plt.savefig(pdf_filename)
    print(f"Saved latent t-SNE plot as {pdf_filename}")
    plt.close()
    
    base_silhouette = silhouette_score(base_latents, base_labels)
    gcad_silhouette = silhouette_score(gcad_latents, gcad_labels)
    print(f"Base Model Silhouette Score: {base_silhouette:.3f}")
    print(f"GCAD Hyperbolic Encoder Silhouette Score: {gcad_silhouette:.3f}")


def experiment3(train_dataset, train_loader, val_loader, device, num_epochs=10, test_mode=False):
    """
    Experiment 3: Evaluate both methods under a low-data regime with primarily linear noise.
    Also includes an ablation study (GCADModelAblation) and measures reconstruction quality
    with LPIPS alongside MSE/PSNR/SSIM.
    
    Args:
        train_dataset: Training dataset
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run the experiment on
        num_epochs: Number of epochs for training
        test_mode: If True, fewer epochs are run for testing
        
    Returns:
        None (saves plots to logs directory)
    """
    epochs = 1 if test_mode else num_epochs

    subset_indices = list(range(0, len(train_dataset), 5))
    limited_train_dataset = Subset(train_dataset, subset_indices)
    limited_train_loader = DataLoader(limited_train_dataset, batch_size=128, shuffle=True)
    
    print("\n==== Experiment 3: Limited Data Regime with Linear Corruption ====")
    
    print("Training BaseAutoencoder on limited data with linear corruption...")
    base_model_lim = BaseAutoencoder()
    base_model_lim, _ = train_model(base_model_lim, limited_train_loader, device, noise_type="linear", num_epochs=epochs)
    base_mse, base_psnr, base_ssim = evaluate_model(base_model_lim, val_loader, device, noise_type="linear")
    base_lpips = evaluate_lpips(base_model_lim, val_loader, device, noise_type="linear")
    print(f"Base Model (Limited Data): MSE={base_mse:.4f}, PSNR={base_psnr:.2f}, SSIM={base_ssim:.4f}, LPIPS={base_lpips:.4f}")
    
    print("Training GCADModel on limited data with linear corruption...")
    gcad_model_lim = GCADModel()
    gcad_model_lim, _ = train_model(gcad_model_lim, limited_train_loader, device, noise_type="linear", num_epochs=epochs)
    gcad_mse, gcad_psnr, gcad_ssim = evaluate_model(gcad_model_lim, val_loader, device, noise_type="linear")
    gcad_lpips = evaluate_lpips(gcad_model_lim, val_loader, device, noise_type="linear")
    print(f"GCAD Model (Limited Data): MSE={gcad_mse:.4f}, PSNR={gcad_psnr:.2f}, SSIM={gcad_ssim:.4f}, LPIPS={gcad_lpips:.4f}")
    
    print("Training GCAD Ablation Model on limited data with linear corruption...")
    gcad_ablation_model = GCADModelAblation()
    gcad_ablation_model, _ = train_model(gcad_ablation_model, limited_train_loader, device, noise_type="linear", num_epochs=epochs)
    ablation_mse, ablation_psnr, ablation_ssim = evaluate_model(gcad_ablation_model, val_loader, device, noise_type="linear")
    ablation_lpips = evaluate_lpips(gcad_ablation_model, val_loader, device, noise_type="linear")
    print(f"GCAD Ablation Model (Limited Data): MSE={ablation_mse:.4f}, PSNR={ablation_psnr:.2f}, SSIM={ablation_ssim:.4f}, LPIPS={ablation_lpips:.4f}")
    
    methods = ['Base', 'GCAD', 'GCAD Ablation']
    lpips_scores = [base_lpips, gcad_lpips, ablation_lpips]
    plt.figure(figsize=(6,4))
    plt.bar(methods, lpips_scores, color=['blue', 'green', 'red'])
    plt.ylabel("LPIPS Score (Lower is Better)")
    plt.title("LPIPS Comparison Under Limited Data & Linear Corruption")
    plt.tight_layout()
    pdf_filename = "logs/lpips_comparison_linear_pair1.pdf"
    plt.savefig(pdf_filename)
    print(f"Saved LPIPS comparison plot as {pdf_filename}")
    plt.close()


def run_test():
    """
    Quick test function to verify that the code runs.
    It uses a smaller number of epochs and a subsampled dataset so it finishes immediately.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on device: {device}")

    train_dataset, train_loader, _, val_loader = get_cifar10_data(data_dir='./data', batch_size=128)
    
    experiment1(train_loader, val_loader, device, num_epochs=10, test_mode=True)
    
    experiment2(val_loader, device, test_mode=True)
    
    experiment3(train_dataset, train_loader, val_loader, device, num_epochs=10, test_mode=True)
    
    print("\nAll test experiments finished successfully.")


if __name__ == "__main__":
    run_test()
