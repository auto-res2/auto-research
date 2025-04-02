"""
Evaluation module for PriorBrush experiment.

This module implements evaluation metrics and experiments for comparing
SwiftBrush and PriorBrush models.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from preprocess import prepare_for_visualization, compute_ssim


def experiment_inference_and_quality(swift_generator, prior_generator, prompt, 
                                     seed=42, refinement_steps=3, num_runs=5, device="cuda"):
    """
    Run both SwiftBrush and PriorBrush pipelines for a given prompt several times,
    measure inference time and compute quality metrics.
    
    Args:
        swift_generator: SwiftBrush generator instance.
        prior_generator: PriorBrush generator instance.
        prompt (str): Text prompt for image generation.
        seed (int): Base random seed for reproducibility.
        refinement_steps (int): Number of refinement steps for PriorBrush.
        num_runs (int): Number of runs to average metrics over.
        device (str): Device to run the models on.
        
    Returns:
        tuple: (swift_times, prior_times, quality_metrics)
    """
    swift_times = []
    prior_times = []
    quality_metrics = []
    
    print("\n*** Experiment 1: Inference Speed and Image Quality Comparison ***")
    print(f"Running {num_runs} trials for prompt: '{prompt}'")
    
    for i in range(num_runs):
        curr_seed = seed + i  # Using different seeds for multiple trials
        
        start_time = time.time()
        image_swift = swift_generator.generate(prompt, curr_seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_swift = time.time() - start_time
        swift_times.append(elapsed_swift)
        
        start_time = time.time()
        image_prior = prior_generator.generate(prompt, curr_seed, refinement_steps=refinement_steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_prior = time.time() - start_time
        prior_times.append(elapsed_prior)
        
        img_swift_np = prepare_for_visualization(image_swift)
        img_prior_np = prepare_for_visualization(image_prior)
        
        ssim_val = compute_ssim(img_swift_np, img_prior_np)
        quality_metrics.append(ssim_val)
        
        print(f"Trial {i+1}: SwiftBrush time = {elapsed_swift:.4f}s, "
              f"PriorBrush time = {elapsed_prior:.4f}s, SSIM = {ssim_val:.4f}")
    
    swift_mean = np.mean(swift_times)
    swift_std = np.std(swift_times)
    prior_mean = np.mean(prior_times)
    prior_std = np.std(prior_times)
    ssim_mean = np.mean(quality_metrics)
    
    print("\n[Summary for Experiment 1]")
    print(f"SwiftBrush Inference Time: Mean = {swift_mean:.4f}s, Std = {swift_std:.4f}s")
    print(f"PriorBrush Inference Time: Mean = {prior_mean:.4f}s, Std = {prior_std:.4f}s")
    print(f"Average SSIM between outputs: {ssim_mean:.4f}")
    
    return swift_times, prior_times, quality_metrics


def experiment_ablation(swift_generator, prior_generator, prompt, 
                        seed=42, refinement_steps=3, device="cuda", output_path="logs/ablation_study_small.pdf"):
    """
    Perform an ablation study comparing output images with and without the diffusion-based refinement.
    Generate one sample image for each method and visualize a side-by-side comparison along with an error map.
    
    Args:
        swift_generator: SwiftBrush generator instance.
        prior_generator: PriorBrush generator instance.
        prompt (str): Text prompt for image generation.
        seed (int): Random seed for reproducibility.
        refinement_steps (int): Number of refinement steps for PriorBrush.
        device (str): Device to run the models on.
        output_path (str): Path to save the output plot.
        
    Returns:
        float: SSIM value between refined and non-refined images.
    """
    print("\n*** Experiment 2: Ablation Study on the Refinement Stage ***")
    print(f"Evaluating ablation with prompt: '{prompt}'")
    
    image_with_refinement = prior_generator.generate(prompt, seed, refinement_steps=refinement_steps)
    
    image_without_refinement = swift_generator.generate(prompt, seed)
    
    img_with = prepare_for_visualization(image_with_refinement)
    img_without = prepare_for_visualization(image_without_refinement)
    
    error_map = np.abs(img_with - img_without)
    
    ssim_val = compute_ssim(img_with, img_without)
    print(f"Computed SSIM between refined and non-refined images: {ssim_val:.4f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(img_without, 0, 1))
    plt.title("Without Refinement\n(SwiftBrush)")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(img_with, 0, 1))
    plt.title(f"With Refinement\n(PriorBrush, {refinement_steps} steps)")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(error_map, 0, 1))
    plt.title("Error Map (Absolute Diff)")
    plt.axis("off")
    
    plt.suptitle(f"Ablation Study\nSSIM = {ssim_val:.4f}", fontsize=16)
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Ablation study plot saved as '{output_path}'")
    plt.close()
    
    return ssim_val


def experiment_sensitivity(swift_generator, prior_generator, prompt, 
                           seed=42, step_range=[2, 3, 5], device="cuda", 
                           output_path="logs/sensitivity_analysis_small.pdf"):
    """
    Investigate the sensitivity of the image quality and inference time to the number of refinement steps.
    For each value in step_range, run PriorBrush and compare inference time and quality metric.
    
    Args:
        swift_generator: SwiftBrush generator instance.
        prior_generator: PriorBrush generator instance.
        prompt (str): Text prompt for image generation.
        seed (int): Random seed for reproducibility.
        step_range (list): Different refinement steps to test.
        device (str): Device to run the models on.
        output_path (str): Path to save the output plot.
        
    Returns:
        tuple: (time_results, quality_results)
    """
    print("\n*** Experiment 3: Sensitivity Analysis of Refinement Sampling Steps ***")
    print(f"Evaluating sensitivity across refinement steps {step_range} for prompt: '{prompt}'")
    
    time_results = {}
    quality_results = {}
    
    image_base = swift_generator.generate(prompt, seed)
    img_base = prepare_for_visualization(image_base)
    
    for steps in step_range:
        start_time = time.time()
        image_refined = prior_generator.generate(prompt, seed, refinement_steps=steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        time_results[steps] = elapsed_time
        
        img_refined = prepare_for_visualization(image_refined)
        quality_val = compute_ssim(img_refined, img_base)
        quality_results[steps] = quality_val
        
        print(f"Refinement Steps: {steps} --> Inference Time = {elapsed_time:.4f}s, "
              f"SSIM vs. SwiftBrush = {quality_val:.4f}")
    
    steps_list = list(time_results.keys())
    times_list = [time_results[s] for s in steps_list]
    quality_list = [quality_results[s] for s in steps_list]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps_list, times_list, marker='o', linestyle='-', color='blue')
    plt.xlabel("Diffusion Refinement Steps")
    plt.ylabel("Inference Time (s)")
    plt.title("Inference Time vs. Refinement Steps")
    
    plt.subplot(1, 2, 2)
    plt.plot(steps_list, quality_list, marker='o', linestyle='-', color='green')
    plt.xlabel("Diffusion Refinement Steps")
    plt.ylabel("SSIM (vs. SwiftBrush)")
    plt.title("Image Quality vs. Refinement Steps")
    
    plt.suptitle(f"Sensitivity Analysis for prompt: '{prompt}'", fontsize=16)
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Sensitivity analysis plot saved as '{output_path}'")
    plt.close()
    
    return time_results, quality_results


def run_quick_test(swift_generator, prior_generator, config):
    """
    Run a quick test to ensure the code executes correctly.
    
    Args:
        swift_generator: SwiftBrush generator instance.
        prior_generator: PriorBrush generator instance.
        config (dict): Configuration dictionary.
        
    Returns:
        bool: True if the test completes successfully.
    """
    print("\n=== Running Quick Test ===")
    
    test_prompt = config["quick_test"]["prompt"]
    refinement_steps = config["quick_test"]["refinement_steps"]
    step_range = config["quick_test"]["step_range"]
    
    experiment_inference_and_quality(
        swift_generator, prior_generator,
        test_prompt, seed=100, refinement_steps=refinement_steps, num_runs=1
    )
    
    experiment_ablation(
        swift_generator, prior_generator,
        test_prompt, seed=100, refinement_steps=refinement_steps
    )
    
    experiment_sensitivity(
        swift_generator, prior_generator,
        test_prompt, seed=100, step_range=step_range
    )
    
    print("Quick test completed.\n")
    return True
