"""
Evaluation module for TCPGS experiments.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time


def compute_FID(generated_images, ground_truth_dataset=None):
    """
    Compute Fréchet Inception Distance (FID) score.
    
    For this implementation, we use a simplified dummy FID calculation.
    In a real-world scenario, a proper FID implementation would be used.
    
    Args:
        generated_images: Batch of generated images
        ground_truth_dataset: Reference dataset (unused in this dummy implementation)
        
    Returns:
        float: Dummy FID score
    """
    # In a real experiment, compute Fréchet Inception Distance using a proper implementation.
    # Here we return a dummy value based on the mean pixel differences
    gen_mean = generated_images.mean().item()
    # For demonstration we assume the ground truth has mean 0.5
    fid = abs(gen_mean - 0.5) * 100  
    return fid


def experiment_robustness(models, dataloader, device=torch.device("cuda"), noise_levels=None):
    """
    Experiment 1: Robustness to Corrupted Data and Variable Noise Levels
    
    Args:
        models: Dictionary of models to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
        noise_levels: Dictionary of noise types and levels
        
    Returns:
        dict: Results of the experiment
    """
    print("\n=== Experiment 1: Robustness to Corrupted Data and Variable Noise Levels ===")
    
    if noise_levels is None:
        noise_levels = {
            'Gaussian': [0.1, 0.2], 
            'SaltPepper': [0.05, 0.1]
        }
    
    # Get one batch from the loader
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    results = {}
    
    # Process Gaussian noise variants
    for std in noise_levels['Gaussian']:
        gaussian_images = add_gaussian_noise(images, std=std)
        for name, model in models.items():
            model.to(device)
            # For demonstration, we use one denoising step
            denoised = model.denoise_step(gaussian_images, step=0, total_steps=1)
            # Using dummy FID
            fid_score = compute_FID(denoised)
            results[(name, f"Gaussian_{std}")] = fid_score
            print(f"[Gaussian Noise] Model: {name}, Noise STD: {std}, Dummy FID: {fid_score:.2f}")
    
    # Process Salt-and-Pepper noise variants
    for amount in noise_levels['SaltPepper']:
        sp_images = add_salt_pepper_noise(images, amount=amount)
        for name, model in models.items():
            model.to(device)
            denoised = model.denoise_step(sp_images, step=0, total_steps=1)
            fid_score = compute_FID(denoised)
            results[(name, f"SaltPepper_{amount}")] = fid_score
            print(f"[Salt-Pepper Noise] Model: {name}, Amount: {amount}, Dummy FID: {fid_score:.2f}")
    
    # Save visualization for the first batch
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 2)
        plt.title('Gaussian Noise (std=0.2)')
        plt.imshow(add_gaussian_noise(images, std=0.2)[0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 3)
        plt.title('Salt-and-Pepper (amount=0.1)')
        plt.imshow(add_salt_pepper_noise(images, amount=0.1)[0].permute(1, 2, 0).cpu().numpy())
        plt.savefig('logs/noise_examples.png')
        print("Noise examples visualization saved to logs/noise_examples.png")
    except Exception as e:
        print(f"Could not save visualization: {e}")
    
    return results


def experiment_convergence(models, device=torch.device("cuda"), step_counts=None):
    """
    Experiment 2: Convergence Efficiency (Fewer Sampling Steps)
    
    Args:
        models: Dictionary of models to evaluate
        device: Device to run evaluation on
        step_counts: List of step counts to evaluate
        
    Returns:
        dict: Results of the experiment
    """
    print("\n=== Experiment 2: Convergence Efficiency (Fewer Sampling Steps) ===")
    
    if step_counts is None:
        step_counts = [100, 50, 25]
    
    def sample_images(model, num_steps, initial_noise, device):
        """Run the sampling process for a specific number of steps."""
        images = initial_noise.clone().to(device)
        step_times = []
        for step in range(num_steps):
            start_time = time.time()
            images = model.denoise_step(images, step, num_steps)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            step_times.append(time.time() - start_time)
        return images, step_times
    
    initial_noise = get_initial_noise(num_samples=64, device=device)
    results = {}
    
    for steps in step_counts:
        for name, model in models.items():
            model.to(device)
            generated_images, step_times = sample_images(model, steps, initial_noise, device)
            fid_score = compute_FID(generated_images)
            avg_time = np.mean(step_times)
            results[(name, steps)] = {'FID': fid_score, 'AvgTime': avg_time}
            print(f"Model: {name}, Sampling Steps: {steps}, Dummy FID: {fid_score:.2f}, "
                  f"Avg time per step: {avg_time:.4f} sec")
            
            # Save a sample image for the shortest step count
            if steps == min(step_counts):
                try:
                    plt.figure(figsize=(6, 6))
                    plt.title(f'{name} - {steps} steps')
                    plt.imshow(generated_images[0].permute(1, 2, 0).cpu().detach().numpy())
                    plt.savefig(f'logs/{name}_{steps}_steps.png')
                    print(f"Sample image saved to logs/{name}_{steps}_steps.png")
                except Exception as e:
                    print(f"Could not save sample image: {e}")
    
    return results


def experiment_ablation(model_variants, dataloader, device=torch.device("cuda")):
    """
    Experiment 3: Analysis of Gradient Estimation via Tweedie Consistency Correction
    
    Args:
        model_variants: Dictionary of model variants to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
        
    Returns:
        dict: Results of the experiment
    """
    print("\n=== Experiment 3: Tweedie Consistency Correction Ablation Study ===")
    
    # Get one batch from the dataloader
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    ablation_results = {}
    
    for variant_name, model in model_variants.items():
        model.to(device)
        # Add gaussian noise for the experiment
        noisy_images = add_gaussian_noise(images, std=0.2)
        denoised, gradients = model.denoise_with_grad(noisy_images)
        mse_loss = F.mse_loss(denoised, images).item()
        ablation_results[variant_name] = mse_loss
        print(f"Variant: {variant_name}, MSE Loss: {mse_loss:.4f}")
        
        try:
            # Visualize gradient for the first image in the batch
            grad_img = gradients[0].cpu().detach().numpy()  # shape (3, H, W)
            # For visualization, we take the gradient magnitude across channels
            grad_magnitude = np.sqrt(np.sum(grad_img**2, axis=0))
            
            plt.figure(figsize=(10, 4))
            
            # Plot original, noisy, and denoised images
            plt.subplot(1, 4, 1)
            plt.title('Original')
            plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            
            plt.subplot(1, 4, 2)
            plt.title('Noisy')
            plt.imshow(noisy_images[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.axis('off')
            
            plt.subplot(1, 4, 3)
            plt.title('Denoised')
            plt.imshow(denoised[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.axis('off')
            
            plt.subplot(1, 4, 4)
            plt.title('Gradient Magnitude')
            plt.imshow(grad_magnitude, cmap='viridis')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            plt.savefig(f'logs/ablation_{variant_name}.png')
            print(f"Ablation visualization saved to logs/ablation_{variant_name}.png")
            
            # Additional visualization for gradient field
            plt.figure(figsize=(8, 8))
            plt.title(f"Gradient Field: {variant_name}")
            
            # Use first two channels for vector field
            U = grad_img[0]
            V = grad_img[1]
            H, W = U.shape
            X, Y = np.meshgrid(np.arange(0, W, 2), np.arange(0, H, 2))
            U = U[::2, ::2]
            V = V[::2, ::2]
            
            plt.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', scale=1)
            plt.xlabel("Width")
            plt.ylabel("Height")
            plt.gca().invert_yaxis()
            plt.savefig(f'logs/gradient_field_{variant_name}.png')
            print(f"Gradient field visualization saved to logs/gradient_field_{variant_name}.png")
            
        except Exception as e:
            print(f"Could not save ablation visualization: {e}")
    
    return ablation_results


# Import here to avoid circular imports
from src.preprocess import add_gaussian_noise, add_salt_pepper_noise, get_initial_noise
