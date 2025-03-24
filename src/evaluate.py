# src/evaluate.py
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from src.utils.models import AmbientDiffusionModel, OneStepGenerator
from src.utils.data import get_dataloaders
from src.utils.metrics import measure_inference_time, compute_memory_usage
from src.utils.experiments import experiment2_noise_robustness, experiment3_ablation_study

def experiment1_efficiency_benchmark(
    ambient_model_path=None,
    one_step_model_path=None,
    device='cuda',
    num_samples=100,
    diffusion_steps=50,
    batch_size=1,
    save_dir='./logs'
):
    """
    Experiment 1: Efficiency and Inference-Time Benchmarking
    Compares the inference time and memory usage of the ambient diffusion model
    vs. the one-step generator.
    """
    print("\n========== Experiment 1: Efficiency and Inference-Time Benchmarking ==========")
    
    # Create logs directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize models
    ambient_model = AmbientDiffusionModel().to(device)
    one_step_model = OneStepGenerator().to(device)
    
    # Load model weights if provided
    if ambient_model_path and os.path.exists(ambient_model_path):
        ambient_model.load_state_dict(torch.load(ambient_model_path))
        print(f"Loaded ambient model from {ambient_model_path}")
    
    if one_step_model_path and os.path.exists(one_step_model_path):
        one_step_model.load_state_dict(torch.load(one_step_model_path))
        print(f"Loaded one-step model from {one_step_model_path}")
    
    # Set up input shape for CIFAR10 (B, C, H, W)
    latent_shape = (batch_size, 3, 32, 32)
    
    # Measure inference time
    ambient_time = measure_inference_time(
        ambient_model, latent_shape, num_samples, steps=diffusion_steps, device=device
    )
    one_step_time = measure_inference_time(
        one_step_model, latent_shape, num_samples, steps=1, device=device
    )
    
    print(f"Ambient Diffusion inference time per sample (ms): {ambient_time:.2f}")
    print(f"One-Step Generator inference time per sample (ms): {one_step_time:.2f}")
    print(f"Speedup factor: {ambient_time / one_step_time:.2f}x")
    
    # Measure memory usage
    ambient_memory = compute_memory_usage(ambient_model, latent_shape, steps=diffusion_steps, device=device)
    one_step_memory = compute_memory_usage(one_step_model, latent_shape, steps=1, device=device)
    
    print(f"Ambient Diffusion memory usage (MB): {ambient_memory:.2f}")
    print(f"One-Step Generator memory usage (MB): {one_step_memory:.2f}")
    
    # Avoid division by zero
    if one_step_memory > 0 and ambient_memory > 0:
        print(f"Memory reduction: {ambient_memory / one_step_memory:.2f}x")
    else:
        print("Memory reduction: N/A (memory usage too small to measure accurately)")
    
    # Generate and save sample images
    ambient_model.eval()
    one_step_model.eval()
    
    with torch.no_grad():
        # Generate samples from both models
        noise = torch.randn(16, 3, 32, 32, device=device)
        
        # Ambient diffusion (multi-step)
        ambient_samples = noise.clone()
        for step in range(diffusion_steps):
            ambient_samples = ambient_model(ambient_samples, step)
        
        # One-step generator
        one_step_samples = one_step_model(noise)
    
    # Convert to displayable format (0-1 range)
    ambient_samples = torch.clamp(ambient_samples, 0, 1)
    one_step_samples = torch.clamp(one_step_samples, 0, 1)
    
    # Plot and save results
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Samples Comparison: Ambient Diffusion vs One-Step Generator')
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # Ambient diffusion samples (left)
        ambient_img = ambient_samples[i].cpu().permute(1, 2, 0).numpy()
        axes[row, col].imshow(ambient_img)
        axes[row, col].set_title('Ambient' if i < 4 else '')
        axes[row, col].axis('off')
        
        # One-step generator samples (right)
        one_step_img = one_step_samples[i].cpu().permute(1, 2, 0).numpy()
        axes[row, col + 4].imshow(one_step_img)
        axes[row, col + 4].set_title('One-Step' if i < 4 else '')
        axes[row, col + 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experiment1_samples_comparison.png'))
    print(f"Saved sample comparison to {os.path.join(save_dir, 'experiment1_samples_comparison.png')}")
    
    # Return results for reporting
    results = {
        'ambient_time': ambient_time,
        'one_step_time': one_step_time,
        'speedup_factor': ambient_time / one_step_time,
        'ambient_memory': ambient_memory,
        'one_step_memory': one_step_memory
    }
    
    # Avoid division by zero
    if one_step_memory > 0 and ambient_memory > 0:
        results['memory_reduction'] = ambient_memory / one_step_memory
    else:
        results['memory_reduction'] = 0.0  # Default value when measurement is too small
        
    return results

def run_all_experiments(
    device='cuda',
    save_dir='./logs',
    models_dir='./models',
    noise_levels=[0.1, 0.3, 0.6]
):
    """
    Run all three experiments and collect results.
    """
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Run Experiment 1: Efficiency Benchmark
    exp1_results = experiment1_efficiency_benchmark(
        device=device,
        num_samples=50,  # Reduced for testing
        diffusion_steps=50,
        save_dir=save_dir
    )
    
    # Run Experiment 2: Noise Robustness
    exp2_results = experiment2_noise_robustness(
        noise_levels=noise_levels,
        batch_size=32,
        epochs=1,
        device=device,
        save_dir=save_dir
    )
    
    # Run Experiment 3: Ablation Study
    exp3_results = experiment3_ablation_study(
        device=device,
        num_synthetic=50,  # Reduced for testing
        batch_size=16,
        epochs=1,
        save_dir=save_dir
    )
    
    # Compile and return all results
    all_results = {
        'experiment1': exp1_results,
        'experiment2': exp2_results,
        'experiment3': exp3_results
    }
    
    return all_results

if __name__ == "__main__":
    # Simple test run of the evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run experiment 1 only for a quick test
    results = experiment1_efficiency_benchmark(
        device=device,
        num_samples=10,  # Reduced for testing
        diffusion_steps=10,  # Reduced for testing
        save_dir='./logs'
    )
    
    print("\nExperiment 1 Results Summary:")
    print(f"Speedup factor: {results['speedup_factor']:.2f}x")
    print(f"Memory reduction: {results['memory_reduction']:.2f}x")
