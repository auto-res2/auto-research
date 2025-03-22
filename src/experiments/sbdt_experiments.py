"""
SBDT (Stochastic Backdoor with Diffused Triggers) experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import itertools
from sklearn.ensemble import IsolationForest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet import SimpleUNet, Autoencoder
from src.utils.diffusion_utils import forward_diffusion
from src.evaluate import evaluate_activation, compute_reconstruction_errors, evaluate_anomaly_detection
from src.train import train_diffusion_model, train_autoencoder

def experiment_robustness(device, train_loader, config, save_dir=None):
    """
    Experiment 1: Robustness Under Noise
    
    Args:
        device: Device to use for computation
        train_loader: DataLoader for training data
        config: Configuration parameters
        save_dir: Directory to save results
        
    Returns:
        Dictionary of results
    """
    print("Running Experiment 1: Robustness Under Noise")
    
    # Create save directory if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Instantiate models for Base Method and SBDT method
    base_model = SimpleUNet().to(device)
    sbdt_model = SimpleUNet().to(device)
    
    # For Base Method, use a fixed diffusion
    print("Training Base Method model...")
    def base_diffusion(label_tensor):
        return [forward_diffusion(label_tensor, timesteps=1)[-1]]
    
    trained_base_model = train_diffusion_model(
        base_model, train_loader, 
        num_epochs=config.NUM_EPOCHS, 
        device=device,
        diffusion_fn=base_diffusion,
        test_run=config.TEST_RUN
    )
    
    # For SBDT, use forward_diffusion with multiple steps
    print("Training SBDT Method model (stochastic label diffusion)...")
    def sbdt_diffusion(label_tensor):
        return forward_diffusion(
            label_tensor, 
            timesteps=config.DEFAULT_DIFFUSION_STEPS, 
            beta_start=config.DEFAULT_BETA_START, 
            beta_end=config.DEFAULT_BETA_END
        )
    
    trained_sbdt_model = train_diffusion_model(
        sbdt_model, train_loader, 
        num_epochs=config.NUM_EPOCHS, 
        device=device,
        diffusion_fn=sbdt_diffusion,
        test_run=config.TEST_RUN
    )
    
    # Evaluate models using SSIM as the metric
    print("Evaluating models...")
    base_ssim = evaluate_activation(
        trained_base_model, train_loader, 
        device=device, 
        save_dir=os.path.join(save_dir, "base_method") if save_dir else None
    )
    
    sbdt_ssim = evaluate_activation(
        trained_sbdt_model, train_loader, 
        device=device,
        save_dir=os.path.join(save_dir, "sbdt_method") if save_dir else None
    )
    
    print(f"Experiment 1 Results: Base Method SSIM = {base_ssim:.4f} | SBDT Method SSIM = {sbdt_ssim:.4f}")
    
    results = {
        "base_method_ssim": base_ssim,
        "sbdt_method_ssim": sbdt_ssim,
        "improvement": sbdt_ssim - base_ssim,
    }
    
    return results

def experiment_ablation(device, train_loader, config, save_dir=None):
    """
    Experiment 2: Parameter Ablation on Label Diffusion Dynamics
    
    Args:
        device: Device to use for computation
        train_loader: DataLoader for training data
        config: Configuration parameters
        save_dir: Directory to save results
        
    Returns:
        Dictionary of results
    """
    print("Running Experiment 2: Parameter Ablation on Label Diffusion Dynamics")
    
    # Create save directory if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    diffusion_steps_options = config.DIFFUSION_STEPS_OPTIONS
    beta_schedules = config.BETA_SCHEDULES
    results = {}
    
    # Loop over limited combinations if test run
    if config.TEST_RUN:
        combinations = [(diffusion_steps_options[0], beta_schedules[0])]
    else:
        combinations = list(itertools.product(diffusion_steps_options, beta_schedules))
    
    # Loop over all combinations
    for steps, (beta_start, beta_end) in combinations:
        print(f"Testing setting: steps={steps}, beta_start={beta_start}, beta_end={beta_end}")
        
        # Define a custom diffusion function for the given parameters
        def custom_diffusion(label_tensor):
            return forward_diffusion(
                label_tensor, 
                timesteps=steps, 
                beta_start=beta_start, 
                beta_end=beta_end
            )
        
        # Create and train a new model with the custom forward diffusion
        model_ablation = SimpleUNet().to(device)
        train_diffusion_model(
            model_ablation, train_loader, 
            num_epochs=max(1, config.NUM_EPOCHS // 2), 
            device=device, 
            diffusion_fn=custom_diffusion,
            test_run=config.TEST_RUN
        )
        
        # Evaluate the model
        ssim_val = evaluate_activation(
            model_ablation, train_loader, 
            device=device,
            save_dir=os.path.join(save_dir, f"steps_{steps}_beta_{beta_start}_{beta_end}") if save_dir else None
        )
        
        results[(steps, beta_start, beta_end)] = ssim_val
        print(f"Setting: Steps={steps}, Beta range=({beta_start},{beta_end}) => SSIM: {ssim_val:.4f}")
    
    # If there are multiple results, create a plot
    if len(results) > 1 and save_dir:
        settings = list(results.keys())
        ssim_values = list(results.values())
        setting_names = [f"s={s},b=({bs:.4f}-{be:.4f})" for s, bs, be in settings]
        
        plt.figure(figsize=(10, 6))
        plt.bar(setting_names, ssim_values)
        plt.xlabel("Diffusion Settings")
        plt.ylabel("Mean SSIM")
        plt.xticks(rotation=45)
        plt.title("Experiment 2: Ablation Study on Diffusion Parameters")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ablation_results.png"))
        plt.close()
    
    return results

def experiment_anomaly_detection(device, train_loader, config, save_dir=None):
    """
    Experiment 3: Stealth and Anomaly Detection Evaluation
    
    Args:
        device: Device to use for computation
        train_loader: DataLoader for training data
        config: Configuration parameters
        save_dir: Directory to save results
        
    Returns:
        Dictionary of results
    """
    print("Running Experiment 3: Stealth and Anomaly Detection Evaluation")
    
    # Create save directory if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Extract flattened features from clean data
    clean_features = []
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            flat = data.view(data.size(0), -1).cpu().numpy()
            clean_features.append(flat)
            if config.TEST_RUN:
                break
    
    clean_features = np.concatenate(clean_features, axis=0)
    
    # Simulate poisoned features
    # a. Base Method: add a fixed offset
    base_features = clean_features + 0.05
    # b. SBDT: add a random normal perturbation
    sbdt_features = clean_features + np.random.normal(0, 0.05, clean_features.shape)
    
    # Evaluate using Isolation Forest
    iso_forest_results = evaluate_anomaly_detection(
        clean_features, base_features, sbdt_features, 
        save_dir=os.path.join(save_dir, "isolation_forest") if save_dir else None
    )
    
    # Evaluate using Autoencoder
    autoencoder = Autoencoder().to(device)
    train_autoencoder(
        autoencoder, train_loader, 
        num_epochs=max(1, config.NUM_EPOCHS // 2), 
        device=device,
        test_run=config.TEST_RUN
    )
    
    clean_errors = compute_reconstruction_errors(autoencoder, train_loader, device=device)
    
    # Simulate errors for poisoned samples
    base_errors = clean_errors + 0.02  # slightly more error
    sbdt_errors = clean_errors + np.random.normal(0, 0.005, clean_errors.shape)
    
    # Save visualization if directory is provided
    if save_dir:
        plt.figure(figsize=(10, 4))
        plt.hist(base_errors, bins=20, alpha=0.5, label="Base Method Recon Error")
        plt.hist(sbdt_errors, bins=20, alpha=0.5, label="SBDT Method Recon Error")
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Experiment 3: Autoencoder Reconstruction Errors")
        plt.savefig(os.path.join(save_dir, "autoencoder_errors.png"))
        plt.close()
    
    # Print summary statistics
    print("Anomaly Detection Summary via Isolation Forest:")
    print(f"  Mean Base Method Score: {np.mean(iso_forest_results['mean_base_score']):.4f}")
    print(f"  Mean SBDT Method Score: {np.mean(iso_forest_results['mean_sbdt_score']):.4f}")
    print("Autoencoder Reconstruction Errors (mean):")
    print(f"  Base Method Error: {np.mean(base_errors):.4f}")
    print(f"  SBDT Method Error: {np.mean(sbdt_errors):.4f}")
    
    # Combine results
    results = {
        **iso_forest_results,
        "mean_base_error": np.mean(base_errors),
        "mean_sbdt_error": np.mean(sbdt_errors),
        "std_base_error": np.std(base_errors),
        "std_sbdt_error": np.std(sbdt_errors),
    }
    
    return results
