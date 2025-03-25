import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def evaluate_resolution_generalization(model, dataloaders, config, device='cuda'):
    """
    Evaluate the model's ability to handle different resolutions
    
    Args:
        model: Trained ATBFNPipeline model
        dataloaders: Dictionary of dataloaders at different resolutions
        config: Configuration dictionary
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    results = {}
    
    print("\n--- Evaluating Resolution Generalization ---")
    for resolution, loader in dataloaders.items():
        print(f"\nTesting on resolution: {resolution}x{resolution}")
        
        # Get a batch of images
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device)
        
        with torch.no_grad():
            outputs = model(imgs)
            
        # Calculate metrics
        complexity_map = outputs['complexity_map']
        reconstructed = outputs['reconstructed']
        
        # Resize original images to 32x32 to match reconstruction
        imgs_resized = F.interpolate(imgs, size=(32, 32))
        
        # Calculate reconstruction MSE
        mse = F.mse_loss(reconstructed, imgs_resized).item()
        
        # Calculate complexity map statistics
        complexity_stats = {
            'min': complexity_map.min().item(),
            'max': complexity_map.max().item(),
            'mean': complexity_map.mean().item(),
            'std': complexity_map.std().item()
        }
        
        print(f"Reconstruction MSE: {mse:.4f}")
        print(f"Complexity map stats: min={complexity_stats['min']:.4f}, "
              f"max={complexity_stats['max']:.4f}, mean={complexity_stats['mean']:.4f}")
        
        results[resolution] = {
            'mse': mse,
            'complexity_stats': complexity_stats
        }
    
    return results

def evaluate_token_evolution(model, dataloader, config, device='cuda'):
    """
    Evaluate token evolution effectiveness
    
    Args:
        model: ATBFNPipeline model
        dataloader: DataLoader for test data
        config: Configuration dictionary
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    print("\n--- Evaluating Token Evolution ---")
    
    # Get a batch of images
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    
    # Test with different numbers of evolution steps
    step_counts = [1, 3, 5, 10] if not config.get('quick_test', False) else [1, 3]
    results = {}
    
    for steps in step_counts:
        print(f"\nTesting with {steps} evolution steps")
        
        with torch.no_grad():
            outputs = model(imgs, num_steps=steps)
            
        reconstructed = outputs['reconstructed']
        
        # Resize original images to match reconstruction
        imgs_resized = F.interpolate(imgs, size=(32, 32))
        
        # Calculate reconstruction MSE
        mse = F.mse_loss(reconstructed, imgs_resized).item()
        
        print(f"Reconstruction MSE with {steps} steps: {mse:.4f}")
        
        results[steps] = {
            'mse': mse
        }
    
    return results

def evaluate_attention_impact(config, device='cuda'):
    """
    Compare performance with and without attention
    
    Args:
        config: Configuration dictionary
        device: Device to run evaluation on
        
    Returns:
        Dictionary of comparative metrics
    """
    from train import ATBFNPipeline
    from preprocess import get_dataloader
    
    print("\n--- Evaluating Impact of Cross-Token Attention ---")
    
    # Create test dataloader
    dataloader = get_dataloader(config, train=False)
    
    # Get a batch of images
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    
    # Create models with and without attention
    print("Creating model WITH attention")
    model_with_attention = ATBFNPipeline(config, token_dim=config.get('token_dim', 64), use_attention=True).to(device)
    
    print("Creating model WITHOUT attention")
    model_without_attention = ATBFNPipeline(config, token_dim=config.get('token_dim', 64), use_attention=False).to(device)
    
    results = {}
    
    # Evaluate both models
    with torch.no_grad():
        print("Running inference with attention")
        output_with = model_with_attention(imgs)
        reconstructed_with = output_with['reconstructed']
        
        print("Running inference without attention")
        output_without = model_without_attention(imgs)
        reconstructed_without = output_without['reconstructed']
    
    # Resize original images to match reconstruction
    imgs_resized = F.interpolate(imgs, size=(32, 32))
    
    # Calculate metrics
    mse_with = F.mse_loss(reconstructed_with, imgs_resized).item()
    mse_without = F.mse_loss(reconstructed_without, imgs_resized).item()
    
    # Calculate difference between outputs
    output_diff_norm = torch.norm(reconstructed_with - reconstructed_without).item()
    
    print(f"MSE with attention: {mse_with:.4f}")
    print(f"MSE without attention: {mse_without:.4f}")
    print(f"Output difference norm: {output_diff_norm:.4f}")
    
    results = {
        'mse_with_attention': mse_with,
        'mse_without_attention': mse_without,
        'output_difference_norm': output_diff_norm
    }
    
    return results

def run_full_evaluation(model, dataloaders, config, device='cuda'):
    """
    Run all evaluation metrics
    
    Args:
        model: Trained ATBFNPipeline model
        dataloaders: Dictionary of dataloaders at different resolutions
        config: Configuration dictionary
        device: Device to run evaluation on
        
    Returns:
        Dictionary of all evaluation results
    """
    results = {
        'resolution_generalization': evaluate_resolution_generalization(model, dataloaders, config, device),
        'token_evolution': evaluate_token_evolution(model, dataloaders[config['image_resolutions'][0]], config, device),
        'attention_impact': evaluate_attention_impact(config, device)
    }
    
    return results

def quick_test_evaluation(model, dataloader, config, device='cuda'):
    """
    Quick test to verify model functionality
    
    Args:
        model: ATBFNPipeline model
        dataloader: DataLoader for test data
        config: Configuration dictionary
        device: Device to run evaluation on
    """
    print("\n--- Quick Test Evaluation ---")
    
    model.eval()
    
    # Get a batch of images
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    
    with torch.no_grad():
        outputs = model(imgs)
    
    reconstructed = outputs['reconstructed']
    
    # Resize original images to match reconstruction
    imgs_resized = F.interpolate(imgs, size=(32, 32))
    
    # Calculate reconstruction MSE
    mse = F.mse_loss(reconstructed, imgs_resized).item()
    
    print(f"Test complete. Reconstruction MSE: {mse:.4f}")
    print(f"Model output shapes:")
    print(f"  - Tokens shape: {outputs['tokens'].shape}")
    print(f"  - Complexity map shape: {outputs['complexity_map'].shape}")
    print(f"  - Reconstructed shape: {outputs['reconstructed'].shape}")
    
    return mse
