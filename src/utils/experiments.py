# src/utils/experiments.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from .models import AmbientDiffusionModel, OneStepGenerator
from .data import get_dataloaders
from .metrics import measure_inference_time, compute_memory_usage

def experiment2_noise_robustness(
    noise_levels=[0.1, 0.3, 0.6],
    batch_size=32,
    epochs=1,
    device='cuda',
    save_dir='./logs'
):
    """
    Experiment 2: Robustness to Noise and Memorization Prevention
    Trains models on datasets with different noise levels and evaluates
    the score identity loss evolution.
    """
    print("\n========== Experiment 2: Robustness to Noise and Memorization Prevention ==========")
    
    # Create logs directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    for sigma in noise_levels:
        print(f"\n-- Training on dataset with Gaussian noise sigma = {sigma:.1f} --")
        
        # Get dataloaders for this noise level
        train_loader, _ = get_dataloaders(batch_size=batch_size, noise_level=sigma)
        
        # Initialize models
        ambient_model = AmbientDiffusionModel().to(device)
        one_step_model = OneStepGenerator().to(device)
        
        # Initialize optimizer
        optimizer = optim.Adam(one_step_model.parameters(), lr=1e-3)
        
        # Initialize loss function
        mse_loss = nn.MSELoss()
        
        # Training loop
        score_align_losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (noisy_imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                noisy_imgs = noisy_imgs.to(device)
                
                # Obtain target scores from the ambient diffusion model (teacher)
                with torch.no_grad():
                    target_scores = ambient_model(noisy_imgs, step=25)
                
                # Get one-step generator output
                one_step_output = one_step_model(noisy_imgs)
                
                # Compute score identity loss
                loss = mse_loss(one_step_output, target_scores)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                # For test purposes, break after a few batches
                if batch_idx >= 2:
                    break
            
            avg_loss = np.mean(epoch_losses)
            score_align_losses.append(avg_loss)
            print(f"Epoch {epoch+1}: Mean Score Identity Loss {avg_loss:.4f}")
        
        # Save results for this noise level
        results[sigma] = score_align_losses
        
        # Plot the loss evolution for this noise level
        plt.figure()
        plt.plot(score_align_losses, marker='o')
        plt.title(f"Score Identity Loss Evolution (sigma={sigma:.1f})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(save_dir, f"experiment2_loss_sigma_{sigma:.1f}.png"))
        plt.close()
        print(f"Saved loss plot for sigma={sigma:.1f}")
    
    return results

def experiment3_ablation_study(
    device='cuda',
    num_synthetic=100,
    batch_size=16,
    epochs=1,
    save_dir='./logs'
):
    """
    Experiment 3: Data-Free Distillation Efficacy and Ablation Study
    Performs an ablation study on the components of the ASD method.
    """
    print("\n========== Experiment 3: Data-Free Distillation Efficacy and Ablation Study ==========")
    
    # Create logs directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize ambient diffusion model
    ambient_model = AmbientDiffusionModel().to(device)
    
    # Generate synthetic dataset using the ambient model
    print(f"Generating synthetic dataset with {num_synthetic} samples...")
    synthetic_inputs = []
    ambient_targets = []
    
    for i in range(num_synthetic):
        latent = torch.randn(1, 3, 32, 32, device=device)
        with torch.no_grad():
            # Use a fixed step to generate a pseudo-ground truth sample
            target = ambient_model(latent, step=30)
        synthetic_inputs.append(latent.cpu())
        ambient_targets.append(target.cpu())
    
    synthetic_inputs = torch.cat(synthetic_inputs, dim=0)
    ambient_targets = torch.cat(ambient_targets, dim=0)
    
    # Define loss configurations for ablation study
    configs = {
        'full': {'score_identity': True, 'consistency': True},
        'no_consistency': {'score_identity': True, 'consistency': False},
        'no_score_identity': {'score_identity': False, 'consistency': True}
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nTraining with configuration: {config_name}")
        
        # Initialize one-step generator
        one_step_model = OneStepGenerator().to(device)
        
        # Initialize optimizer
        optimizer = optim.Adam(one_step_model.parameters(), lr=1e-3)
        
        # Initialize loss function
        mse_loss = nn.MSELoss()
        
        # Training loop
        losses_log = []
        
        for epoch in range(epochs):
            # Create dataloader for this epoch
            indices = torch.randperm(len(synthetic_inputs))[:batch_size * 3]  # Limit to 3 batches for testing
            batch_inputs = synthetic_inputs[indices].to(device)
            batch_targets = ambient_targets[indices].to(device)
            
            for i in range(0, len(indices), batch_size):
                # Get batch
                inputs = batch_inputs[i:i+batch_size]
                targets = batch_targets[i:i+batch_size]
                
                # Forward pass
                outputs = one_step_model(inputs)
                
                # Compute losses according to configuration
                loss_components = {}
                
                if config.get('score_identity', True):
                    loss_components['score_identity'] = mse_loss(outputs, targets)
                
                if config.get('consistency', True):
                    # Example consistency loss: require that a slight perturbation in the output remains consistent
                    perturbation = torch.randn_like(inputs) * 0.05
                    perturbed_outputs = one_step_model(inputs + perturbation)
                    loss_components['consistency'] = mse_loss(outputs, perturbed_outputs)
                
                # Total loss
                loss = sum(loss_components.values())
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses_log.append(loss.item())
                
                # Print loss components
                loss_str = f"Batch {i//batch_size + 1}: Total Loss {loss.item():.4f}"
                for name, value in loss_components.items():
                    loss_str += f", {name} {value.item():.4f}"
                print(loss_str)
        
        # Save final loss for this configuration
        final_loss = losses_log[-1] if losses_log else float('inf')
        results[config_name] = {'final_loss': final_loss}
        print(f"Configuration '{config_name}' final loss: {final_loss:.4f}")
    
    # Plot comparison of final losses
    plt.figure(figsize=(10, 6))
    config_names = list(results.keys())
    final_losses = [results[name]['final_loss'] for name in config_names]
    
    plt.bar(config_names, final_losses)
    plt.title("Ablation Study: Final Loss Comparison")
    plt.ylabel("Final Loss")
    plt.savefig(os.path.join(save_dir, "experiment3_ablation_comparison.png"))
    plt.close()
    print("Saved ablation study comparison plot")
    
    return results
