"""
Evaluation module for MCAD experiments.
Handles model evaluation, metrics calculation, and visualization.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def calculate_psnr(reconstructed, original):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = torch.mean((reconstructed - original) ** 2).item()
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse)))
    return psnr.item()

def evaluate_model(model, dataloader, device, noise_std=0.1, max_batches=None):
    """Evaluate model performance on a dataset."""
    model.eval()
    mse_loss = nn.MSELoss()
    
    total_loss = 0.0
    total_psnr = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for idx, (data, _) in enumerate(dataloader):
            # Break after max_batches if specified (for test runs)
            if max_batches is not None and idx >= max_batches:
                break
                
            # Move data to device
            data = data.to(device)
            
            # Add noise to simulate corruption
            corrupted = data + torch.randn_like(data) * noise_std
            
            # Generate reconstruction
            reconstruction = model(corrupted)
            
            # Calculate metrics
            loss = mse_loss(reconstruction, data).item()
            psnr = calculate_psnr(reconstruction, data)
            
            # Update totals
            total_loss += loss
            total_psnr += psnr
            batch_count += 1
    
    # Calculate averages
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_psnr = total_psnr / batch_count if batch_count > 0 else 0
    
    return {
        'loss': avg_loss,
        'psnr': avg_psnr
    }

def visualize_reconstructions(model, dataloader, device, noise_std=0.1, num_samples=5):
    """Visualize original, corrupted, and reconstructed images."""
    model.eval()
    
    # Get a batch of data
    data, _ = next(iter(dataloader))
    
    # Select a few samples
    data = data[:num_samples].to(device)
    
    # Add noise to simulate corruption
    corrupted = data + torch.randn_like(data) * noise_std
    
    # Generate reconstructions
    with torch.no_grad():
        reconstructions = model(corrupted)
    
    # Move tensors to CPU and convert to numpy
    data = data.cpu().numpy()
    corrupted = corrupted.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    
    for i in range(num_samples):
        # Original images
        axes[0, i].imshow(np.transpose(data[i], (1, 2, 0)))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Corrupted images
        axes[1, i].imshow(np.clip(np.transpose(corrupted[i], (1, 2, 0)), 0, 1))
        axes[1, i].set_title("Corrupted")
        axes[1, i].axis('off')
        
        # Reconstructed images
        axes[2, i].imshow(np.transpose(reconstructions[i], (1, 2, 0)))
        axes[2, i].set_title("Reconstructed")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("logs/reconstructions.png")
    plt.close()
    
    return "logs/reconstructions.png"

def compare_models(models_dict, dataloader, device, noise_std=0.1, max_batches=None):
    """Compare multiple models on the same dataset."""
    results = {}
    
    for name, model in models_dict.items():
        print(f"Evaluating {name} model...")
        model_results = evaluate_model(
            model, 
            dataloader, 
            device, 
            noise_std=noise_std,
            max_batches=max_batches
        )
        results[name] = model_results
        print(f"  {name}: Loss = {model_results['loss']:.4f}, PSNR = {model_results['psnr']:.2f} dB")
    
    return results
