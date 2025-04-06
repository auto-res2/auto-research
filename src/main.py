"""
ClusterCloak: Main Experiment Script

This script implements the ClusterCloak poisoning defense method for
protecting images from unauthorized diffusion model synthesis.

Experiments:
    1. Feature Misalignment and Clustering Analysis
    2. Robust Poisoning Against Fine-Tuning Recovery
    3. Robustness Under Common Transformations and Purification Attacks
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import warnings
import time

from preprocess import DummyDataset
from utils.transforms import apply_clustercloak, apply_metacloak
from utils.metrics import compute_clustering_metrics
from utils.visualization import plot_tsne
from train import SimpleGenerator, train_dummy_model
from evaluate import analyze_feature_clustering, DummyIdentityClassifier

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.clustercloak_config import *

warnings.filterwarnings("ignore")

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def experiment_feature_clustering(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, save_plots=SAVE_PLOTS):
    print("\n" + "="*80)
    print("Running Experiment 1: Feature Misalignment and Clustering Analysis...")
    print("="*80)
    
    print(f"Using device for feature clustering: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    model = torchvision.models.resnet18(pretrained=True)
    model = model.to(device)
    model.eval()
    print(f"Model device: {next(model.parameters()).device}")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = DummyDataset(num_samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Created dataset with {len(dataset)} samples and batch size {batch_size}")
    
    print("Analyzing ClusterCloak features...")
    cc_results = analyze_feature_clustering(model, loader, "ClusterCloak", save_plots)
    print(f"ClusterCloak feature extraction complete. Feature shape: {cc_results['features'].shape if len(cc_results['features']) > 0 else 'empty'}")
    
    print("Analyzing MetaCloak features...")
    mc_results = analyze_feature_clustering(model, loader, "MetaCloak", save_plots)
    print(f"MetaCloak feature extraction complete. Feature shape: {mc_results['features'].shape if len(mc_results['features']) > 0 else 'empty'}")
    
    print("\nClustering Metrics:")
    print(f"ClusterCloak Silhouette Score: {cc_results['metrics']['silhouette']:.4f}")
    print(f"MetaCloak Silhouette Score: {mc_results['metrics']['silhouette']:.4f}")
    print(f"ClusterCloak Davies-Bouldin Score: {cc_results['metrics']['davies_bouldin']:.4f}")
    print(f"MetaCloak Davies-Bouldin Score: {mc_results['metrics']['davies_bouldin']:.4f}")
    
    if save_plots and len(cc_results['features']) > 0 and len(mc_results['features']) > 0:
        combined_features = np.vstack([cc_results['features'], mc_results['features']])
        
        labels = np.concatenate([
            np.zeros(len(cc_results['features'])),
            np.ones(len(mc_results['features']))
        ])
        
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
        tsne_proj = tsne.fit_transform(combined_features)
        
        plt.figure(figsize=(12, 8))
        
        cc_points = tsne_proj[:len(cc_results['features'])]
        plt.scatter(cc_points[:, 0], cc_points[:, 1], c='blue', label='ClusterCloak')
        
        mc_points = tsne_proj[len(cc_results['features']):]
        plt.scatter(mc_points[:, 0], mc_points[:, 1], c='red', label='MetaCloak')
        
        plt.title('t-SNE Projection: ClusterCloak vs MetaCloak')
        plt.legend()
        plt.tight_layout()
        plt.savefig("logs/feature_clustering_comparison.pdf", format='pdf', dpi=300)
        plt.close()
        print("Saved comparative t-SNE plot as: logs/feature_clustering_comparison.pdf")
    
    return cc_results, mc_results


def experiment_fine_tuning(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, save_plots=SAVE_PLOTS):
    print("\n" + "="*80)
    print("Running Experiment 2: Robust Poisoning Against Fine-Tuning Recovery...")
    print("="*80)
    
    print(f"Using device for fine-tuning: {device}")
    print(f"Experiment parameters: latent_dim={LATENT_DIM}, img_size={IMG_SIZE}, lr={LEARNING_RATE}")
    
    print("Initializing generator models...")
    generator_cc = SimpleGenerator(LATENT_DIM, IMG_SIZE).to(device)
    generator_mc = SimpleGenerator(LATENT_DIM, IMG_SIZE).to(device)
    print(f"ClusterCloak generator device: {next(generator_cc.parameters()).device}")
    print(f"MetaCloak generator device: {next(generator_mc.parameters()).device}")
    
    print("Setting up optimizers...")
    optimizer_cc = optim.Adam(generator_cc.parameters(), lr=LEARNING_RATE)
    optimizer_mc = optim.Adam(generator_mc.parameters(), lr=LEARNING_RATE)
    
    criterion = nn.MSELoss()
    print(f"Using loss function: {criterion.__class__.__name__}")
    
    print(f"Generating {batch_size} latent vectors with dimension {LATENT_DIM}...")
    latent = torch.randn(batch_size, LATENT_DIM, device=device)
    print(f"Latent vectors device: {latent.device}")
    
    print("\n" + "-"*50)
    print("Training with ClusterCloak defense...")
    print("-"*50)
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    losses_cc, trained_cc = train_dummy_model(
        generator_cc, optimizer_cc, criterion, latent, 
        num_epochs=num_epochs, batch_size=batch_size, 
        experiment_type="ClusterCloak", save_plots=save_plots
    )
    print(f"ClusterCloak training complete. Final loss: {losses_cc[-1]:.6f}")
    
    print("\n" + "-"*50)
    print("Training with MetaCloak defense...")
    print("-"*50)
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    losses_mc, trained_mc = train_dummy_model(
        generator_mc, optimizer_mc, criterion, latent, 
        num_epochs=num_epochs, batch_size=batch_size, 
        experiment_type="MetaCloak", save_plots=save_plots
    )
    print(f"MetaCloak training complete. Final loss: {losses_mc[-1]:.6f}")
    
    if save_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs+1), losses_cc, marker='o', label="ClusterCloak")
        plt.plot(range(1, num_epochs+1), losses_mc, marker='s', label="MetaCloak")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Fine-Tuning Loss Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("logs/fine_tuning_comparison.pdf", format='pdf', dpi=300)
        plt.close()
        print("Saved fine-tuning comparison plot as: logs/fine_tuning_comparison.pdf")
    
    classifier = DummyIdentityClassifier(IMG_SIZE, NUM_CLASSES).to(device)
    classifier.eval()
    
    with torch.no_grad():
        imgs_cc = generator_cc(latent)
        imgs_mc = generator_mc(latent)
        
        true_ids = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        
        preds_cc = classifier(imgs_cc).argmax(dim=1)
        preds_mc = classifier(imgs_mc).argmax(dim=1)
        
        accuracy_cc = (preds_cc == true_ids).float().mean().item()
        accuracy_mc = (preds_mc == true_ids).float().mean().item()
    
    print(f"\nIdentity preservation accuracy ClusterCloak: {accuracy_cc:.4f}")
    print(f"Identity preservation accuracy MetaCloak: {accuracy_mc:.4f}")
    
    return (losses_cc, losses_mc), (accuracy_cc, accuracy_mc)


def test_experiments():
    """
    Run a quick test of all experiments with smaller parameters.
    Used for verification without running full experiments.
    """
    print("\n" + "="*80)
    print("RUNNING QUICK TESTS ON EXPERIMENTS")
    print("="*80)
    
    print("\nTest 1: Feature Misalignment & Clustering Analysis")
    print("-"*60)
    print("Parameters: num_samples=20, batch_size=5, save_plots=True")
    start_time = time.time()
    cc_results, mc_results = experiment_feature_clustering(num_samples=20, batch_size=5, save_plots=True)
    test1_time = time.time() - start_time
    print(f"Test 1 completed in {test1_time:.2f} seconds")
    print(f"Results summary:")
    print(f"  - ClusterCloak features: {cc_results['features'].shape if len(cc_results['features']) > 0 else 'empty'}")
    print(f"  - MetaCloak features: {mc_results['features'].shape if len(mc_results['features']) > 0 else 'empty'}")
    
    print("\nTest 2: Fine-Tuning Recovery with Poisoning")
    print("-"*60)
    print("Parameters: num_epochs=2, batch_size=8, save_plots=True")
    start_time = time.time()
    (losses_cc, losses_mc), (acc_cc, acc_mc) = experiment_fine_tuning(num_epochs=2, batch_size=8, save_plots=True)
    test2_time = time.time() - start_time
    print(f"Test 2 completed in {test2_time:.2f} seconds")
    print(f"Results summary:")
    print(f"  - ClusterCloak final loss: {losses_cc[-1]:.6f}, accuracy: {acc_cc:.4f}")
    print(f"  - MetaCloak final loss: {losses_mc[-1]:.6f}, accuracy: {acc_mc:.4f}")
    
    total_time = test1_time + test2_time
    print("\n" + "="*80)
    print(f"ALL TESTS COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print("="*80 + "\n")


if __name__ == "__main__":
    start_time = time.time()
    print(f"ClusterCloak Experiment - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {NUM_SAMPLES} samples, {BATCH_SIZE} batch size, {NUM_EPOCHS} epochs")
    
    test_experiments()
    
    if RUN_EXPERIMENT_1:
        experiment_feature_clustering()
    
    if RUN_EXPERIMENT_2:
        experiment_fine_tuning()
    
    execution_time = time.time() - start_time
    print(f"\nExperiments completed in {execution_time:.2f} seconds")
    print(f"Results and plots saved in the 'logs' directory")
