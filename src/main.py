#!/usr/bin/env python
"""
Main script for running Tweedie-Guided Global Consistent Video Editing (TG-GCVE) experiments.

This script orchestrates the entire experimental pipeline:
1. Data preprocessing
2. Model training with different variants
3. Model evaluation and visualization
4. Results reporting

The experiments focus on three key aspects of TG-GCVE:
1. Tweedie-inspired consistency loss
2. Global context aggregation for global editing
3. Adaptive fusion and iterative refinement
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# Import from other modules
from preprocess import set_seed, preprocess_for_training, add_noise
from train import (
    TGGCVEModule, 
    train_tg_gcve, 
    train_variant, 
    visualize_global_context,
    experiment_consistency_loss_ablation,
    experiment_global_context_evaluation,
    experiment_adaptive_fusion_refinement
)
from evaluate import (
    evaluate_model, 
    evaluate_variants, 
    visualize_editing_results,
    evaluate_adaptive_fusion
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if CUDA is available and print GPU info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU available, using CPU")

# --------------------------
# Configuration
# --------------------------

def load_config(config_path=None):
    """
    Load configuration from a JSON file or use default configuration.
    
    Args:
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    # Default configuration
    default_config = {
        'data_dir': 'data/sample_videos',
        'models_dir': 'models',
        'logs_dir': 'logs',
        'batch_size': 2,
        'frame_size': (256, 256),
        'sequence_length': 16,
        'epochs': 5,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'noise_level': 0.1,
        'perceptual_weight': 0.1,
        'consistency_weight': 0.1,
        'global_editing_intensity': 1.0,
        'print_freq': 10,
        'save_freq': 1,
        'target_text': 'edited video',
        'use_consistency_loss': True,
        'seed': 42,
        'quick_test': True,
        'experiments': ['consistency', 'global_context', 'adaptive_fusion'],
        'variants': {
            'full_TG_GCVE': {
                'use_refined_stage': True, 
                'use_consistency_loss': True, 
                'use_global_context': True
            },
            'single_stage': {
                'use_refined_stage': False, 
                'use_consistency_loss': False, 
                'use_global_context': True
            },
            'no_consistency': {
                'use_refined_stage': True, 
                'use_consistency_loss': False, 
                'use_global_context': True
            },
            'no_global_context': {
                'use_refined_stage': True, 
                'use_consistency_loss': True, 
                'use_global_context': False
            },
            'iterative_refinement': {
                'use_refined_stage': True, 
                'use_consistency_loss': True, 
                'use_global_context': True,
                'iterative_refinement': True,
                'num_iterations': 3
            }
        }
    }
    
    # Load configuration from file if provided
    if config_path is not None and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Update default config with file config
                for key, value in file_config.items():
                    if key == 'variants' and 'variants' in default_config:
                        # Merge variants
                        for variant_key, variant_value in value.items():
                            default_config['variants'][variant_key] = variant_value
                    else:
                        default_config[key] = value
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            print("Using default configuration")
    else:
        print("No configuration file provided or file not found")
        print("Using default configuration")
    
    return default_config

# --------------------------
# Experiment Functions
# --------------------------

def setup_directories(config):
    """
    Create necessary directories for the experiment.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Updated configuration dictionary
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(config['data_dir']):
        os.makedirs(config['data_dir'], exist_ok=True)
        print(f"Created directory: {config['data_dir']}")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(config['models_dir']):
        os.makedirs(config['models_dir'], exist_ok=True)
        print(f"Created directory: {config['models_dir']}")
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(config['logs_dir']):
        os.makedirs(config['logs_dir'], exist_ok=True)
        print(f"Created directory: {config['logs_dir']}")
    
    # Create experiment timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config['logs_dir'], f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created experiment directory: {experiment_dir}")
    
    # Update config with experiment directory
    config['experiment_dir'] = experiment_dir
    
    return config

def create_dummy_data(config):
    """
    Create dummy data for testing.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (dummy_input, dummy_gt, dummy_dataloader)
    """
    print("\nCreating dummy data for testing...")
    
    # Create dummy input and ground truth with smaller dimensions for testing
    batch_size = 2
    seq_len = 4  # Reduced from default sequence_length
    frame_size = (64, 64)  # Reduced from default frame_size
    channels = 3
    
    # Create dummy input tensor [B, T, C, H, W]
    dummy_input = torch.randn(batch_size, seq_len, channels, *frame_size).to(device)
    
    # Create dummy ground truth tensor [B, T, C, H, W]
    dummy_gt = torch.randn(batch_size, seq_len, channels, *frame_size).to(device)
    
    # Create dummy dataloader
    dummy_dataset = TensorDataset(
        dummy_input.view(-1, channels, *frame_size),
        dummy_gt.view(-1, channels, *frame_size)
    )
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size)
    
    # Create dummy batch for dataloader
    dummy_batch = {
        'frames': dummy_input,
        'noisy_frames': add_noise(dummy_input.view(-1, channels, *frame_size), 0.1).view(batch_size, seq_len, channels, *frame_size)
    }
    
    print(f"Created dummy data with shape: {dummy_input.shape}")
    
    return dummy_input, dummy_gt, dummy_dataloader, dummy_batch

def quick_test(config):
    """
    Run a quick test of all experiments to confirm code execution.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if test passed, False otherwise
    """
    print("\n=== Running quick test for all experiments ===")
    
    try:
        # Create dummy data
        dummy_input, dummy_gt, dummy_dataloader, dummy_batch = create_dummy_data(config)
        
        # Experiment 1 Quick Test: Consistency Loss Ablation
        print("\n[Experiment 1 Quick Test: Consistency Loss Ablation]")
        test_variant = {"use_refined_stage": True, "use_consistency_loss": True, "use_global_context": True}
        model = train_variant(test_variant, dummy_dataloader, epochs=1)
        
        # Experiment 2 Quick Test: Global Context Evaluation
        print("\n[Experiment 2 Quick Test: Global Context Evaluation]")
        global_model = TGGCVEModule(
            use_refined_stage=True,
            use_consistency_loss=True,
            use_global_context=True
        ).to(device)
        visualize_global_context(global_model, dummy_input[0, 0].unsqueeze(0))
        
        # Experiment 3 Quick Test: Adaptive Fusion and Refinement
        print("\n[Experiment 3 Quick Test: Adaptive Fusion and Refinement]")
        refinement_model = TGGCVEModule(
            use_refined_stage=True,
            use_consistency_loss=True,
            use_global_context=True,
            iterative_refinement=True,
            num_iterations=1
        ).to(device)
        
        # Forward pass with iterative refinement
        with torch.no_grad():
            output, _, _, _, intermediates = refinement_model(dummy_input[0, 0].unsqueeze(0))
            print(f"Output shape: {output.shape}")
            print(f"Number of intermediate outputs: {len(intermediates) if intermediates else 0}")
        
        print("\nQuick test completed successfully.")
        return True
    
    except Exception as e:
        print(f"\nQuick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_experiment_consistency_loss(config):
    """
    Run Experiment 1: Tweedie-inspired consistency loss ablation.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Experiment results
    """
    print("\n" + "="*80)
    print("Experiment 1: Tweedie-Inspired Consistency Loss Ablation")
    print("="*80)
    
    # Run experiment
    models = experiment_consistency_loss_ablation(
        data_dir=config['data_dir'],
        epochs=config['epochs']
    )
    
    # Evaluate models
    variants = {
        "full_TG_GCVE": config['variants']['full_TG_GCVE'],
        "single_stage": config['variants']['single_stage'],
        "no_consistency": config['variants']['no_consistency']
    }
    
    results = evaluate_variants(
        variants=variants,
        data_dir=config['data_dir'],
        target_text=config['target_text']
    )
    
    # Save results
    results_path = os.path.join(config['experiment_dir'], 'consistency_loss_results.json')
    with open(results_path, 'w') as f:
        # Convert tensor values to float for JSON serialization
        serializable_results = {}
        for variant, metrics in results.items():
            serializable_results[variant] = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        json.dump(serializable_results, f, indent=4)
    
    print(f"Saved consistency loss experiment results to {results_path}")
    
    return results

def run_experiment_global_context(config):
    """
    Run Experiment 2: Global context aggregation evaluation.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Experiment results
    """
    print("\n" + "="*80)
    print("Experiment 2: Global Context Aggregation Evaluation")
    print("="*80)
    
    # Run experiment
    models = experiment_global_context_evaluation(
        data_dir=config['data_dir'],
        epochs=config['epochs']
    )
    
    # Evaluate models
    variants = {
        "with_global_context": config['variants']['full_TG_GCVE'],
        "without_global_context": config['variants']['no_global_context']
    }
    
    results = evaluate_variants(
        variants=variants,
        data_dir=config['data_dir'],
        target_text=config['target_text']
    )
    
    # Save results
    results_path = os.path.join(config['experiment_dir'], 'global_context_results.json')
    with open(results_path, 'w') as f:
        # Convert tensor values to float for JSON serialization
        serializable_results = {}
        for variant, metrics in results.items():
            serializable_results[variant] = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        json.dump(serializable_results, f, indent=4)
    
    print(f"Saved global context experiment results to {results_path}")
    
    return results

def run_experiment_adaptive_fusion(config):
    """
    Run Experiment 3: Adaptive fusion and iterative refinement.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Experiment results
    """
    print("\n" + "="*80)
    print("Experiment 3: Adaptive Fusion and Iterative Refinement")
    print("="*80)
    
    # Run experiment
    models = experiment_adaptive_fusion_refinement(
        data_dir=config['data_dir'],
        epochs=config['epochs']
    )
    
    # Evaluate adaptive fusion
    results = evaluate_adaptive_fusion(
        data_dir=config['data_dir'],
        target_text=config['target_text']
    )
    
    # Save results
    results_path = os.path.join(config['experiment_dir'], 'adaptive_fusion_results.json')
    with open(results_path, 'w') as f:
        # Convert tensor values to float for JSON serialization
        serializable_results = {
            'results': {},
            'fusion_improvement': {k: float(v) for k, v in results['fusion_improvement'].items()},
            'refinement_improvement': {k: float(v) for k, v in results['refinement_improvement'].items()}
        }
        
        for variant, metrics in results['results'].items():
            serializable_results['results'][variant] = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        
        json.dump(serializable_results, f, indent=4)
    
    print(f"Saved adaptive fusion experiment results to {results_path}")
    
    return results

def run_all_experiments(config):
    """
    Run all experiments.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: All experiment results
    """
    results = {}
    
    # Run quick test if enabled
    if config.get('quick_test', True):
        quick_test_passed = quick_test(config)
        if not quick_test_passed:
            print("\nQuick test failed. Exiting.")
            return None
            
        # Quick test completed successfully
        print("\nQuick test completed successfully.")
        # Full experiments will continue unless specified otherwise
    
    # Run experiments based on configuration
    experiments = config.get('experiments', ['consistency', 'global_context', 'adaptive_fusion'])
    
    if 'consistency' in experiments:
        results['consistency'] = run_experiment_consistency_loss(config)
    
    if 'global_context' in experiments:
        results['global_context'] = run_experiment_global_context(config)
    
    if 'adaptive_fusion' in experiments:
        results['adaptive_fusion'] = run_experiment_adaptive_fusion(config)
    
    # Save all results
    all_results_path = os.path.join(config['experiment_dir'], 'all_results.json')
    with open(all_results_path, 'w') as f:
        # Create a serializable version of results
        serializable_results = {}
        for exp_name, exp_results in results.items():
            if exp_name == 'adaptive_fusion':
                serializable_results[exp_name] = {
                    'fusion_improvement': {k: float(v) for k, v in exp_results['fusion_improvement'].items()},
                    'refinement_improvement': {k: float(v) for k, v in exp_results['refinement_improvement'].items()}
                }
            else:
                serializable_results[exp_name] = {}
                for variant, metrics in exp_results.items():
                    serializable_results[exp_name][variant] = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        
        json.dump(serializable_results, f, indent=4)
    
    print(f"\nSaved all experiment results to {all_results_path}")
    
    return results

def print_summary(results, config):
    """
    Print a summary of all experiment results.
    
    Args:
        results (dict): All experiment results
        config (dict): Configuration dictionary
    """
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        if key != 'variants':
            print(f"  {key}: {value}")
    
    # Print results summary
    if results is None:
        print("\nNo results to summarize.")
        return
    
    print("\nResults Summary:")
    
    # Consistency Loss Experiment
    if 'consistency' in results:
        print("\n1. Tweedie-Inspired Consistency Loss Ablation:")
        for variant, metrics in results['consistency'].items():
            print(f"  {variant}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    # Global Context Experiment
    if 'global_context' in results:
        print("\n2. Global Context Aggregation Evaluation:")
        for variant, metrics in results['global_context'].items():
            print(f"  {variant}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    # Adaptive Fusion Experiment
    if 'adaptive_fusion' in results:
        print("\n3. Adaptive Fusion and Iterative Refinement:")
        
        # Print fusion improvement
        print("  Adaptive Fusion Improvement over Baseline:")
        for metric, value in results['adaptive_fusion']['fusion_improvement'].items():
            print(f"    {metric}: {value:.2f}%")
        
        # Print refinement improvement
        print("  Iterative Refinement Improvement over Fusion:")
        for metric, value in results['adaptive_fusion']['refinement_improvement'].items():
            print(f"    {metric}: {value:.2f}%")
    
    print("\nExperiment completed successfully.")

# --------------------------
# Main Function
# --------------------------

def main():
    """
    Main function to run the TG-GCVE experiments.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TG-GCVE experiments')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing data')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test only')
    parser.add_argument('--experiment', type=str, default=None, choices=['consistency', 'global_context', 'adaptive_fusion', 'all'], help='Experiment to run')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.data_dir is not None:
        config['data_dir'] = args.data_dir
    if args.quick_test:
        config['quick_test'] = True
    if args.seed is not None:
        config['seed'] = args.seed
    if args.experiment is not None:
        if args.experiment == 'all':
            config['experiments'] = ['consistency', 'global_context', 'adaptive_fusion']
        else:
            config['experiments'] = [args.experiment]
    
    # Setup directories
    config = setup_directories(config)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        if key != 'variants':
            print(f"  {key}: {value}")
    
    # Check if data directory contains video files
    if not os.path.exists(config['data_dir']) or not any(f.endswith(('.mp4', '.avi', '.mov')) for f in os.listdir(config['data_dir'])):
        print(f"\nNo video files found in {config['data_dir']}")
        print("Creating dummy data for testing...")
        
        # Create dummy data directory if it doesn't exist
        os.makedirs(config['data_dir'], exist_ok=True)
        
        # Create dummy video file (just a placeholder)
        dummy_video_path = os.path.join(config['data_dir'], 'dummy_video.mp4')
        with open(dummy_video_path, 'w') as f:
            f.write("This is a dummy video file for testing.")
        
        print(f"Created dummy video file at {dummy_video_path}")
    
    # Run experiments
    if args.quick_test or (config.get('quick_test', True) and not any(exp in config.get('experiments', []) for exp in ['consistency', 'global_context', 'adaptive_fusion'])):
        # Run only quick test
        quick_test_passed = quick_test(config)
        if quick_test_passed:
            print("\nQuick test passed. Exiting.")
        else:
            print("\nQuick test failed. Exiting.")
    else:
        # Run all experiments
        results = run_all_experiments(config)
        
        # Print summary
        print_summary(results, config)
    
    print("\nAll experiments completed. Exiting.")

if __name__ == "__main__":
    main()
