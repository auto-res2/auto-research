import torch
import torch.nn as nn
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from train import GSDModel

def generate_images(model, num_images, batch_size, device):
    """
    Generate images using the one-step GSD model
    
    Args:
        model: Trained GSD model
        num_images: Number of images to generate
        batch_size: Batch size for generation
        device: Device to run generation on
        
    Returns:
        generated_images: Tensor of generated images
    """
    model.eval()
    generated_images = []
    
    with torch.no_grad():
        for _ in range((num_images + batch_size - 1) // batch_size):
            # Sample latent codes from a normal distribution
            curr_batch_size = min(batch_size, num_images - len(generated_images) * batch_size)
            latent_shape = (curr_batch_size, model.config['model']['latent_dim'], 8, 8)  # Adjust shape based on model architecture
            latent = torch.randn(latent_shape).to(device)
            
            # Generate images
            images = model.decoder(latent)
            generated_images.append(images)
            
            if len(generated_images) * batch_size >= num_images:
                break
    
    return torch.cat(generated_images[:num_images // batch_size + 1], dim=0)[:num_images]

def multi_step_generation(model, num_images, batch_size, steps=10, device=None):
    """
    Simulate multi-step generation process for comparison
    
    Args:
        model: Trained GSD model
        num_images: Number of images to generate
        batch_size: Batch size for generation
        steps: Number of steps for multi-step generation
        device: Device to run generation on
        
    Returns:
        generated_images: Tensor of generated images
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    generated_images = []
    
    with torch.no_grad():
        for _ in range((num_images + batch_size - 1) // batch_size):
            curr_batch_size = min(batch_size, num_images - len(generated_images) * batch_size)
            latent_shape = (curr_batch_size, model.config['model']['latent_dim'], 8, 8)  # Adjust shape based on model architecture
            latent = torch.randn(latent_shape).to(device)
            
            # Simulate an iterative refinement process
            for _ in range(steps):
                latent = latent - 0.01 * torch.randn_like(latent)  # Dummy iterative update
            
            # Generate images from refined latent
            images = model.decoder(latent)
            generated_images.append(images)
            
            if len(generated_images) * batch_size >= num_images:
                break
    
    return torch.cat(generated_images[:num_images // batch_size + 1], dim=0)[:num_images]

def interpolate_latents(z1, z2, num_steps=8):
    """
    Linear interpolation between two latent codes
    
    Args:
        z1: First latent code
        z2: Second latent code
        num_steps: Number of interpolation steps
        
    Returns:
        interpolated: Tensor of interpolated latent codes
    """
    alphas = torch.linspace(0, 1, num_steps, device=z1.device)
    interpolated = []
    for alpha in alphas:
        interpolated.append((1 - alpha) * z1 + alpha * z2)
    return torch.cat(interpolated, dim=0)

def perceptual_difference(vgg, img1, img2):
    """
    Compute perceptual difference using a pretrained VGG16
    
    Args:
        vgg: Pretrained VGG16 model
        img1: First image
        img2: Second image
        
    Returns:
        perceptual_diff: Perceptual difference score
    """
    # Extract features from the images
    feat1 = vgg(img1)
    feat2 = vgg(img2)
    return torch.mean((feat1 - feat2)**2)

def evaluate_model(model, config, output_dir=None):
    """
    Evaluate the GSD model with various experiments
    
    Args:
        model: Trained GSD model
        config: Configuration dictionary
        output_dir: Directory to save evaluation results
        
    Returns:
        results: Dictionary containing evaluation results
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    results = {}
    
    # Experiment 1: Ablation Study on Dual-Loss Components
    # This would involve training different models with different loss configurations
    # For this implementation, we just print a placeholder for the demonstration
    print("\n=== Experiment 1: Ablation Study on Dual-Loss Components ===")
    print("Note: For a complete ablation study, multiple models with different loss configurations")
    print("would need to be trained. See the training script for loss implementation.")
    
    # Experiment 2: Evaluation of Inference Speed and Generation Efficiency
    print("\n=== Experiment 2: Inference Speed and Generation Efficiency ===")
    
    # Parameters for generation
    num_images = config['evaluation']['num_samples']
    batch_size = config['evaluation']['batch_size']
    
    # One-step (GSD) generation timing
    start_time = time.time()
    generated_gsd = generate_images(model, num_images, batch_size, device)
    gsd_time = time.time() - start_time
    
    # Multi-step generation timing
    start_time = time.time()
    generated_multi = multi_step_generation(model, num_images, batch_size, steps=10, device=device)
    multi_time = time.time() - start_time
    
    print(f"GSD one-step generation time: {gsd_time:.2f} s")
    print(f"Multi-step generation time: {multi_time:.2f} s")
    print(f"Throughput GSD: {num_images/gsd_time:.2f} images/s, "
          f"Multi-step: {num_images/multi_time:.2f} images/s")
    print(f"Speedup factor: {multi_time/gsd_time:.2f}x")
    
    results['generation_time'] = {
        'gsd': gsd_time,
        'multi_step': multi_time,
        'speedup': multi_time/gsd_time
    }
    
    # Save generated images
    if output_dir is not None:
        # Convert images for visualization
        def convert_images_for_display(images, filename):
            # Denormalize
            images_np = (images.cpu().detach() * 0.5 + 0.5).clamp(0, 1).numpy()
            
            # Create grid of images
            grid_size = int(np.ceil(np.sqrt(len(images_np))))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            
            for i, img in enumerate(images_np):
                if i >= grid_size * grid_size:
                    break
                row, col = i // grid_size, i % grid_size
                ax = axes[row, col] if grid_size > 1 else axes
                # Transform from (C, H, W) to (H, W, C)
                img_display = np.transpose(img, (1, 2, 0))
                ax.imshow(img_display)
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(len(images_np), grid_size * grid_size):
                row, col = i // grid_size, i % grid_size
                if grid_size > 1:
                    axes[row, col].axis('off')
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
        
        # Save the generated images
        convert_images_for_display(generated_gsd[:16], 'gsd_generated.png')
        convert_images_for_display(generated_multi[:16], 'multi_step_generated.png')
    
    # Experiment 3: Latent Space Interpolation and Disentanglement
    print("\n=== Experiment 3: Latent Space Interpolation and Disentanglement ===")
    
    # Generate two random latent vectors
    latent_shape = (1, model.config['model']['latent_dim'], 8, 8)  # Adjust shape based on model architecture
    z1 = torch.randn(latent_shape, device=device)
    z2 = torch.randn(latent_shape, device=device)
    
    # Interpolate between them
    num_steps = 8
    interpolated_images = []
    
    # Generate images from interpolated latents
    with torch.no_grad():
        for alpha in torch.linspace(0, 1, num_steps, device=device):
            # Linear interpolation
            interpolated_latent = (1 - alpha) * z1 + alpha * z2
            # Generate image
            image = model.decoder(interpolated_latent)
            interpolated_images.append(image)
        
        interpolated_images = torch.cat(interpolated_images, dim=0)
    
    # Save interpolation results
    if output_dir is not None:
        # Convert images for visualization
        interp_imgs = (interpolated_images.cpu().detach() * 0.5 + 0.5).clamp(0, 1).numpy()
        
        fig, axes = plt.subplots(1, num_steps, figsize=(20, 3))
        for idx, img in enumerate(interp_imgs):
            img_display = np.transpose(img, (1, 2, 0))
            axes[idx].imshow(img_display)
            axes[idx].axis('off')
        
        plt.suptitle("Latent Space Linear Interpolation")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interpolation.png'))
        plt.close()
    
    # Compute Perceptual Path Length (PPL) using a pretrained VGG16
    print("Computing Perceptual Path Length (PPL)...")
    
    try:
        vgg = models.vgg16(weights='DEFAULT').features.to(device)
        vgg.eval()
        
        # Resize images for VGG if needed
        resize = transforms.Resize((224, 224))
        ppl_values = []
        
        with torch.no_grad():
            for i in range(num_steps - 1):
                img1 = resize(interpolated_images[i].unsqueeze(0))
                img2 = resize(interpolated_images[i+1].unsqueeze(0))
                ppl = perceptual_difference(vgg, img1, img2)
                ppl_values.append(ppl.item())
                
        avg_ppl = np.mean(ppl_values)
        print(f"Average Perceptual Path Length (PPL): {avg_ppl:.4f}")
        
        results['ppl'] = avg_ppl
    except Exception as e:
        print(f"Warning: Could not compute PPL: {e}")
        results['ppl'] = None
    
    return results
