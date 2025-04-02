"""
Main script for running the PTDA experiments.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import prepare_dummy_data
from src.train import train_ptda_model, train_ablated_model, train_baseline_model
from src.evaluate import (
    evaluate_experiment_1,
    evaluate_experiment_2,
    evaluate_experiment_3
)
from src.models.ptda_model import PTDAModel, AblatedPTDAModel, BaselineModel
from config.ptda.config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, EXPERIMENT_CONFIG, PATHS


def setup_directories():
    """
    Set up the necessary directories for the experiments.
    """
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    os.makedirs(PATHS['logs_dir'], exist_ok=True)
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    os.makedirs(PATHS['figures_dir'], exist_ok=True)


def run_experiment_1(device='cuda', train=False):
    """
    Run Experiment 1: Dynamic Background Synthesis and Consistency Testing.
    
    Args:
        device: Device to run the experiment on
        train: Whether to train the models or use pre-trained/dummy models
    """
    print("\n" + "="*80)
    print("Experiment 1: Dynamic Background Synthesis and Consistency Testing")
    print("="*80)
    
    data_dir = PATHS['data_dir']
    video_paths = prepare_dummy_data(
        data_dir,
        num_videos=2,
        num_frames=EXPERIMENT_CONFIG['experiment_1']['num_frames'],
        height=DATA_CONFIG['frame_height'],
        width=DATA_CONFIG['frame_width']
    )
    print(f"Prepared dummy data: {video_paths}")
    
    if train:
        print("Training models for Experiment 1...")
        ptda_model = train_ptda_model(device=device)
        baseline_model = train_baseline_model(device=device)
    else:
        print("Using dummy models for Experiment 1...")
        ptda_model = PTDAModel(
            include_latent=MODEL_CONFIG['include_latent'],
            latent_dim=MODEL_CONFIG['latent_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
        
        baseline_model = BaselineModel(
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
    
    metrics = evaluate_experiment_1(ptda_model, baseline_model, device=device)
    
    return metrics


def run_experiment_2(device='cuda', train=False):
    """
    Run Experiment 2: Latent Variable Integration Ablation Study.
    
    Args:
        device: Device to run the experiment on
        train: Whether to train the models or use pre-trained/dummy models
    """
    print("\n" + "="*80)
    print("Experiment 2: Latent Variable Integration Ablation Study")
    print("="*80)
    
    if train:
        print("Training models for Experiment 2...")
        ptda_model = train_ptda_model(device=device)
        ablated_model = train_ablated_model(device=device)
    else:
        print("Using dummy models for Experiment 2...")
        ptda_model = PTDAModel(
            include_latent=MODEL_CONFIG['include_latent'],
            latent_dim=MODEL_CONFIG['latent_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
        
        ablated_model = AblatedPTDAModel(
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
    
    metrics = evaluate_experiment_2(ptda_model, ablated_model, device=device)
    
    return metrics


def run_experiment_3(device='cuda', train=False):
    """
    Run Experiment 3: Long-Range Temporal Coherence Evaluation.
    
    Args:
        device: Device to run the experiment on
        train: Whether to train the models or use pre-trained/dummy models
    """
    print("\n" + "="*80)
    print("Experiment 3: Long-Range Temporal Coherence Evaluation")
    print("="*80)
    
    if train:
        print("Training models for Experiment 3...")
        ptda_model = train_ptda_model(device=device)
        baseline_model = train_baseline_model(device=device)
    else:
        print("Using dummy models for Experiment 3...")
        ptda_model = PTDAModel(
            include_latent=MODEL_CONFIG['include_latent'],
            latent_dim=MODEL_CONFIG['latent_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
        
        baseline_model = BaselineModel(
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
    
    metrics = evaluate_experiment_3(ptda_model, baseline_model, device=device)
    
    return metrics


def run_smoke_test(device='cuda'):
    """
    Run a quick smoke test of all experiments.
    
    Args:
        device: Device to run the smoke test on
    """
    print("\n" + "="*80)
    print("Running smoke test to verify that the experimental code executes correctly.")
    print("="*80)
    
    EXPERIMENT_CONFIG['experiment_1']['num_frames'] = 5
    EXPERIMENT_CONFIG['experiment_2']['num_samples'] = 2
    EXPERIMENT_CONFIG['experiment_2']['num_frames'] = 3
    EXPERIMENT_CONFIG['experiment_3']['num_frames'] = 5
    
    run_experiment_1(device=device, train=False)
    run_experiment_2(device=device, train=False)
    run_experiment_3(device=device, train=False)
    
    print("\nSmoke test completed successfully.")


def main():
    """
    Main function for running the PTDA experiments.
    """
    parser = argparse.ArgumentParser(description='Run PTDA experiments')
    parser.add_argument('--experiment', type=int, default=0, help='Experiment number (0 for all, 4 for smoke test)')
    parser.add_argument('--train', action='store_true', help='Train models instead of using dummy models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run experiments on (cuda or cpu)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    setup_directories()
    
    print("\nExperiment Configuration:")
    print(f"  Model Config: {MODEL_CONFIG}")
    print(f"  Training Config: {TRAINING_CONFIG}")
    print(f"  Data Config: {DATA_CONFIG}")
    
    start_time = datetime.now()
    print(f"\nStarting experiments at {start_time}")
    
    if args.experiment == 0:
        exp1_metrics = run_experiment_1(device=device, train=args.train)
        exp2_metrics = run_experiment_2(device=device, train=args.train)
        exp3_metrics = run_experiment_3(device=device, train=args.train)
        
        print("\n" + "="*80)
        print("Experiment Summary")
        print("="*80)
        print("Experiment 1 - Dynamic Background Synthesis:")
        print(f"  PTDA SSIM: {exp1_metrics['ptda_ssim']:.4f}, PSNR: {exp1_metrics['ptda_psnr']:.2f}")
        print(f"  Baseline SSIM: {exp1_metrics['baseline_ssim']:.4f}, PSNR: {exp1_metrics['baseline_psnr']:.2f}")
        
        print("\nExperiment 2 - Latent Variable Integration Ablation Study:")
        print(f"  PTDA SSIM: {exp2_metrics['ptda_ssim']:.4f}, PSNR: {exp2_metrics['ptda_psnr']:.2f}")
        print(f"  Ablated SSIM: {exp2_metrics['ablated_ssim']:.4f}, PSNR: {exp2_metrics['ablated_psnr']:.2f}")
        
        print("\nExperiment 3 - Long-Range Temporal Coherence:")
        print(f"  PTDA Consistency: {exp3_metrics['ptda_long_range_consistency']:.4f}")
        print(f"  Baseline Consistency: {exp3_metrics['baseline_long_range_consistency']:.4f}")
    
    elif args.experiment == 1:
        run_experiment_1(device=device, train=args.train)
    
    elif args.experiment == 2:
        run_experiment_2(device=device, train=args.train)
    
    elif args.experiment == 3:
        run_experiment_3(device=device, train=args.train)
    
    elif args.experiment == 4:
        run_smoke_test(device=device)
    
    else:
        print(f"Invalid experiment number: {args.experiment}")
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"\nExecution completed at {end_time}")
    print(f"Total execution time: {execution_time}")


if __name__ == "__main__":
    main()
