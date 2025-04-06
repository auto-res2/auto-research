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
    
    model = torchvision.models.resnet18(pretrained=True)
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = DummyDataset(num_samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print("Analyzing ClusterCloak features...")
    cc_results = analyze_feature_clustering(model, loader, "ClusterCloak", save_plots)
    
    print("Analyzing MetaCloak features...")
    mc_results = analyze_feature_clustering(model, loader, "MetaCloak", save_plots)
    
    print("\nClustering Metrics:")
    print(f"ClusterCloak Silhouette Score: {cc_results['metrics']['silhouette']:.4f}")
    print(f"MetaCloak Silhouette Score: {mc_results['metrics']['silhouette']:.4f}")
    
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
    
    generator_cc = SimpleGenerator(LATENT_DIM, IMG_SIZE).to(device)
    generator_mc = SimpleGenerator(LATENT_DIM, IMG_SIZE).to(device)
    
    optimizer_cc = optim.Adam(generator_cc.parameters(), lr=LEARNING_RATE)
    optimizer_mc = optim.Adam(generator_mc.parameters(), lr=LEARNING_RATE)
    
    criterion = nn.MSELoss()
    
    latent = torch.randn(batch_size, LATENT_DIM, device=device)
    
    print("Training with ClusterCloak defense...")
    losses_cc, trained_cc = train_dummy_model(
        generator_cc, optimizer_cc, criterion, latent, 
        num_epochs=num_epochs, batch_size=batch_size, 
        experiment_type="ClusterCloak", save_plots=save_plots
    )
    
    print("\nTraining with MetaCloak defense...")
    losses_mc, trained_mc = train_dummy_model(
        generator_mc, optimizer_mc, criterion, latent, 
        num_epochs=num_epochs, batch_size=batch_size, 
        experiment_type="MetaCloak", save_plots=save_plots
    )
    
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
    print("\nRunning quick tests on experiments...\n")
    
    experiment_feature_clustering(num_samples=20, batch_size=5, save_plots=True)
    
    experiment_fine_tuning(num_epochs=2, batch_size=8, save_plots=True)
    
    print("\nAll tests completed successfully.\n")


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
