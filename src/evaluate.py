"""
ACSC Evaluation Module

This module implements the three experiments for evaluating the ACSC method:
1. End-to-End Image Quality & Brightness Consistency Evaluation
2. Inference Speed and Computational Efficiency Benchmark
3. Ablation Study: Adaptive Blending and Caching Components
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from utils.acsc_utils import (
    diffusion_process, evaluate_image_quality, 
    diffusion_full_acsc, diffusion_baseline,
    diffusion_process_variant, save_figure
)

def experiment1_image_quality(data):
    """
    Run Experiment 1: Compare the baseline diffusion process with the ACSC-enhanced process.
    Uses a synthetic test image with a horizontal brightness gradient.
    Plots and saves the input and generated images along with printing the quality metrics.
    
    Args:
      data: Dictionary containing datasets
      
    Returns:
      results: Dictionary containing experiment results
    """
    print("\n--- Experiment 1: Image Quality & Brightness Consistency Evaluation ---")
    
    image_tensor = data['datasets']['synthetic_256x256']
    print(f"Synthetic test image shape: {image_tensor.shape}")

    generated_baseline = diffusion_process(image_tensor, acsc_enabled=False)
    generated_acsc = diffusion_process(image_tensor, acsc_enabled=True)

    mae_baseline, ssim_baseline, psnr_baseline = evaluate_image_quality(image_tensor, generated_baseline)
    mae_acsc, ssim_acsc, psnr_acsc = evaluate_image_quality(image_tensor, generated_acsc)

    print(f"Baseline: Brightness MAE: {mae_baseline:.3f}, SSIM: {ssim_baseline:.3f}, PSNR: {psnr_baseline:.3f}")
    print(f"ACSC    : Brightness MAE: {mae_acsc:.3f}, SSIM: {ssim_acsc:.3f}, PSNR: {psnr_acsc:.3f}")

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_tensor.squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(generated_baseline.squeeze(), cmap='gray')
    plt.title("Baseline Diffusion")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(generated_acsc.squeeze(), cmap='gray')
    plt.title("ACSC Diffusion")
    plt.axis('off')
    
    plt.suptitle("Experiment 1: Diffusion Process Comparison")
    pdf_path = save_figure(plt.gcf(), "image_quality_acsc")
    plt.close()

    return {
        'metrics': {
            'baseline': {
                'brightness_mae': mae_baseline,
                'ssim': ssim_baseline,
                'psnr': psnr_baseline,
            },
            'acsc': {
                'brightness_mae': mae_acsc,
                'ssim': ssim_acsc,
                'psnr': psnr_acsc,
            }
        },
        'figures': {
            'comparison': pdf_path
        }
    }

def experiment2_inference_speed(data):
    """
    Run Experiment 2: Benchmark inference speed across different image resolutions.
    For each resolution, run both the baseline and ACSC diffusion process multiple times and average the runtime.
    Plots inference time (in seconds) vs. image resolution.
    
    Args:
      data: Dictionary containing datasets
      
    Returns:
      results: Dictionary containing experiment results
    """
    print("\n--- Experiment 2: Inference Speed & Computational Efficiency Benchmark ---")
    
    datasets = data['datasets']
    resolutions = [128, 256, 512]
    times_acsc = []
    times_baseline = []

    runs_per_res = 5

    for res in resolutions:
        image_key = f'synthetic_{res}x{res}'
        print(f"Evaluating resolution: {res}x{res}")
        
        image = datasets[image_key]
        run_times_acsc = []
        run_times_baseline = []
        
        for _ in range(runs_per_res):
            _, t_acsc = diffusion_full_acsc(image)
            _, t_base = diffusion_baseline(image)
            run_times_acsc.append(t_acsc)
            run_times_baseline.append(t_base)
            
        mean_acsc = np.mean(run_times_acsc)
        mean_baseline = np.mean(run_times_baseline)
        times_acsc.append(mean_acsc)
        times_baseline.append(mean_baseline)
        
        print(f"  Baseline avg time: {mean_baseline:.4f} sec, ACSC avg time: {mean_acsc:.4f} sec")
    
    plt.figure(figsize=(8, 6))
    plt.plot(resolutions, times_baseline, 'r-o', label='Baseline (No Caching)')
    plt.plot(resolutions, times_acsc, 'g-o', label='ACSC (With Adaptive Caching)')
    plt.xlabel('Image Resolution (pixels)')
    plt.ylabel('Average Inference Time (seconds)')
    plt.title('Inference Time Benchmark')
    plt.legend()
    
    pdf_path = save_figure(plt.gcf(), "inference_latency_acsc_vs_baseline")
    plt.close()

    return {
        'metrics': {
            'resolutions': resolutions,
            'baseline_times': times_baseline,
            'acsc_times': times_acsc,
        },
        'figures': {
            'speed_comparison': pdf_path
        }
    }

def experiment3_ablation_study(data):
    """
    Run Experiment 3: Ablation study on the contributions of adaptive blending and caching.
    For a sample image, run three variants: full ACSC, without adaptive blending, and without caching.
    For each run, record quality metrics and inference time.
    Plots a dual bar chart comparing the configurations.
    
    Args:
      data: Dictionary containing datasets
      
    Returns:
      results: Dictionary containing experiment results
    """
    print("\n--- Experiment 3: Ablation Study (Adaptive Blending and Caching) ---")
    
    image = data['datasets']['synthetic_256x256']
    
    configs = {
        "Full ACSC": {"use_caching": True, "adaptive_blending": True},
        "No Adaptive Blending": {"use_caching": True, "adaptive_blending": False},
        "No Caching": {"use_caching": False, "adaptive_blending": True},
    }
    
    results_runtime = {}
    results_quality = {}

    def dummy_quality_metric(image_tensor):
        """
        Dummy quality metric: the mean intensity of the resulting image.
        """
        return image_tensor.mean().item()

    runs = 5
    for label, config in configs.items():
        runtimes = []
        quality_scores = []
        
        for _ in range(runs):
            start = time.time()
            output = diffusion_process_variant(image, **config)
            end = time.time()
            
            runtimes.append(end - start)
            quality_scores.append(dummy_quality_metric(output))
            
        results_runtime[label] = np.mean(runtimes)
        results_quality[label] = np.mean(quality_scores)
        
        print(f"{label}: avg time = {results_runtime[label]:.4f} sec, quality metric = {results_quality[label]:.4f}")
    
    labels = list(configs.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Ablation Configuration')
    ax1.set_ylabel('Inference Time (sec)', color=color1)
    ax1.bar(x - width/2, [results_runtime[l] for l in labels], width, color=color1, label='Inference Time (sec)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Quality Metric (Mean Intensity)', color=color2)
    ax2.bar(x + width/2, [results_quality[l] for l in labels], width, color=color2, label='Quality Metric')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.xticks(x, labels)
    plt.title("Ablation Study: Contribution of Adaptive Blending and Caching")
    fig.tight_layout()
    
    pdf_path = save_figure(plt.gcf(), "ablation_study_acsc")
    plt.close()

    return {
        'metrics': {
            'configurations': labels,
            'runtime': {label: results_runtime[label] for label in labels},
            'quality': {label: results_quality[label] for label in labels},
        },
        'figures': {
            'ablation_comparison': pdf_path
        }
    }

def run_all_experiments(data):
    """
    Run all three experiments and collect results.
    
    Args:
      data: Dictionary containing datasets
      
    Returns:
      results: Dictionary containing all experiment results
    """
    results = {}
    
    results['experiment1'] = experiment1_image_quality(data)
    
    results['experiment2'] = experiment2_inference_speed(data)
    
    results['experiment3'] = experiment3_ablation_study(data)
    
    return results

if __name__ == "__main__":
    from preprocess import preprocess_data
    
    preprocessed_data = preprocess_data()
    results = run_all_experiments(preprocessed_data)
    print("All experiments completed successfully.")
