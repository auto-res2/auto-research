"""
Main script for running the A2Diff experiment.
"""
import os
import time
import torch
import numpy as np
import yaml
from tqdm import tqdm

from preprocess import load_dataset, load_degraded_dataset
from train import init_models, get_schedules, random_dynamic_schedule
from evaluate import (
    generate_samples, 
    generate_samples_variant, 
    generate_samples_with_degradation
)

def load_config():
    """
    Load the experiment configuration.
    
    Returns:
        config: Experiment configuration
    """
    config_path = os.path.join('config', 'a2diff_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def experiment_comparison(diffusion_model, severity_estimator, config):
    """
    Run the comparison experiment (Fixed vs. Adaptive).
    
    Args:
        diffusion_model: Diffusion model
        severity_estimator: Severity estimator network
        config: Experiment configuration
    """
    print("\n====================== Experiment 1: Fixed vs. Adaptive Schedule ======================")
    
    # Load dataset
    dataloader = load_dataset(config, subset_size=16)
    
    print("Running fixed schedule inference...")
    samples_fixed, steps_fixed, times_fixed = generate_samples(
        diffusion_model, 
        dataloader, 
        adaptive=False, 
        severity_estimator=None, 
        config=config
    )
    
    print("Running adaptive schedule inference...")
    samples_adapt, steps_adapt, times_adapt = generate_samples(
        diffusion_model, 
        dataloader, 
        adaptive=True, 
        severity_estimator=severity_estimator, 
        config=config
    )
    
    print("\nExperiment 1 Summary:")
    print(f"Fixed schedule: Avg. steps = {np.mean(steps_fixed):.2f}, Avg. time = {np.mean(times_fixed):.4f} sec")
    print(f"Adaptive schedule: Avg. steps = {np.mean(steps_adapt):.2f}, Avg. time = {np.mean(times_adapt):.4f} sec")
    # In practice, evaluate FID and IS here on samples_fixed and samples_adapt.

def experiment_ablation(diffusion_model, severity_estimator, config):
    """
    Run the ablation study experiment.
    
    Args:
        diffusion_model: Diffusion model
        severity_estimator: Severity estimator network
        config: Experiment configuration
    """
    print("\n====================== Experiment 2: Ablation Study ======================")
    
    # Load dataset
    dataloader = load_dataset(config, subset_size=16)
    
    print("Running full A2Diff variant...")
    samples_full, steps_full, times_full = generate_samples_variant(
        diffusion_model, 
        dataloader, 
        variant='full', 
        severity_estimator=severity_estimator, 
        config=config
    )
    
    print("Running fixed schedule with extra steps variant...")
    samples_fixed_extra, steps_fixed_extra, times_fixed_extra = generate_samples_variant(
        diffusion_model, 
        dataloader, 
        variant='fixed_extra', 
        severity_estimator=None, 
        config=config
    )
    
    print("Running random adaptation variant...")
    samples_random, steps_random, times_random = generate_samples_variant(
        diffusion_model, 
        dataloader, 
        variant='random', 
        severity_estimator=None, 
        config=config
    )
    
    print("\nExperiment 2 Summary:")
    print(f"Full A2Diff: Avg. steps = {np.mean(steps_full):.2f}, Avg. time = {np.mean(times_full):.4f} sec")
    print(f"Fixed Extra: Avg. steps = {np.mean(steps_fixed_extra):.2f}, Avg. time = {np.mean(times_fixed_extra):.4f} sec")
    print(f"Random: Avg. steps = {np.mean(steps_random):.2f}, Avg. time = {np.mean(times_random):.4f} sec")
    # Quality metrics (e.g., FID) would be evaluated here on the produced samples.

def experiment_robustness(diffusion_model, severity_estimator, config):
    """
    Run the robustness evaluation experiment.
    
    Args:
        diffusion_model: Diffusion model
        severity_estimator: Severity estimator network
        config: Experiment configuration
    """
    print("\n====================== Experiment 3: Robustness Evaluation ======================")
    
    # Load degraded dataset
    degraded_dataloader = load_degraded_dataset(config, subset_size=16)
    
    print("Evaluating on degraded dataset using fixed schedule...")
    samples_degraded_fixed, steps_degraded_fixed, times_degraded_fixed, _, flags_fixed = generate_samples_with_degradation(
        diffusion_model, 
        degraded_dataloader, 
        adaptive=False, 
        severity_estimator=None, 
        config=config
    )
    
    print("Evaluating on degraded dataset using adaptive schedule...")
    samples_degraded_adapt, steps_degraded_adapt, times_degraded_adapt, severity_scores_adapt, flags_adapt = generate_samples_with_degradation(
        diffusion_model, 
        degraded_dataloader, 
        adaptive=True, 
        severity_estimator=severity_estimator, 
        config=config
    )
    
    print("\nExperiment 3 Summary:")
    print(f"Fixed schedule on degraded data: Avg. steps = {np.mean(steps_degraded_fixed):.2f}, Avg. time = {np.mean(times_degraded_fixed):.4f} sec")
    if severity_scores_adapt:
        print(f"Adaptive schedule on degraded data: Avg. steps = {np.mean(steps_degraded_adapt):.2f}, Avg. time = {np.mean(times_degraded_adapt):.4f} sec")
        print(f"Adaptive schedule avg. severity scores (per batch): {severity_scores_adapt}")
    # In a full experiment, you would compute quality metrics separately for clean and degraded samples.

def run_experiments():
    """
    Run all experiments.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Initialize models
    diffusion_model, severity_estimator = init_models()
    
    # Run experiments
    if config['experiments']['run_comparison']:
        experiment_comparison(diffusion_model, severity_estimator, config)
    
    if config['experiments']['run_ablation']:
        experiment_ablation(diffusion_model, severity_estimator, config)
    
    if config['experiments']['run_robustness']:
        experiment_robustness(diffusion_model, severity_estimator, config)
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    run_experiments()
