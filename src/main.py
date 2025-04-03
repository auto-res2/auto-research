"""
Main module for running the Latent-Integrated Fingerprint Diffusion (LIFD) experiments.

This script orchestrates the entire process from data preprocessing to model training and evaluation.
It implements three experiments:
1. Dual-Channel Fingerprint Robustness Experiment
2. Ablation Study on Adaptive Balancing
3. Latent Fingerprint Injection Analysis

The results of the experiments are saved as high-quality PDF plots suitable for academic papers.
"""

import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from preprocess import preprocess_data, set_seed
from train import train_model, LIFDModel, FingerprintExtractionNet
from evaluate import evaluate_lifd, dual_channel_fingerprint_experiment, ablation_study_experiment, latent_fingerprint_injection_analysis

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LIFD experiments')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs and plots')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--use_dummy', action='store_true', default=True, 
                        help='Use dummy data instead of real data')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing the data')
    
    parser.add_argument('--fingerprint_dim', type=int, default=128, help='Dimension of the fingerprint')
    parser.add_argument('--num_users', type=int, default=10, help='Number of users')
    parser.add_argument('--adaptive_alpha', type=float, default=0.5, 
                        help='Adaptive balancing parameter (0-1)')
    parser.add_argument('--mode', type=str, default='C', choices=['A', 'B', 'C'], 
                        help='Mode: A=Parameter-Only, B=Latent-Only, C=Dual-Channel')
    
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    parser.add_argument('--skip_training', action='store_true', help='Skip training and load models')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip general evaluation')
    parser.add_argument('--run_exp1', action='store_true', default=True, 
                        help='Run Experiment 1: Dual-Channel Fingerprint Robustness')
    parser.add_argument('--run_exp2', action='store_true', default=True, 
                        help='Run Experiment 2: Ablation Study on Adaptive Balancing')
    parser.add_argument('--run_exp3', action='store_true', default=True, 
                        help='Run Experiment 3: Latent Fingerprint Injection Analysis')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment for the experiments."""
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Using CPU instead.")
        args.device = 'cpu'
        device = torch.device('cpu')
    
    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

def load_or_train_models(args, preprocessed_data):
    """Load pretrained models or train new ones."""
    if args.skip_training:
        print("\nSkipping training and loading pretrained models...")
        
        lifd_model_path = os.path.join(args.save_dir, "lifd_model_final.pth")
        extraction_net_path = os.path.join(args.save_dir, "extraction_net_final.pth")
        
        if not os.path.exists(lifd_model_path) or not os.path.exists(extraction_net_path):
            print("Warning: Pretrained models not found. Training new models...")
            return train_model(preprocessed_data, vars(args))
        
        device = torch.device(args.device)
        lifd_model = LIFDModel(
            fingerprint_dim=args.fingerprint_dim,
            image_size=args.image_size,
            adaptive_alpha=args.adaptive_alpha
        ).to(device)
        
        extraction_net = FingerprintExtractionNet(
            fingerprint_dim=args.fingerprint_dim,
            image_size=args.image_size
        ).to(device)
        
        lifd_model.load_state_dict(torch.load(lifd_model_path, map_location=device))
        extraction_net.load_state_dict(torch.load(extraction_net_path, map_location=device))
        
        lifd_model.set_mode(args.mode)
        
        trained_models = {
            'lifd_model': lifd_model,
            'extraction_net': extraction_net,
            'fingerprints': preprocessed_data['fingerprints'],
            'config': vars(args)
        }
        
        print("Models loaded successfully")
        return trained_models
    else:
        print("\nTraining models...")
        trained_models = train_model(preprocessed_data, vars(args))
        if 'fingerprints' not in trained_models:
            trained_models['fingerprints'] = preprocessed_data['fingerprints']
        return trained_models

def run_experiments(args, trained_models, test_data):
    """Run the specified experiments."""
    results = {}
    
    if not args.skip_evaluation:
        print("\n=== Running General Evaluation ===")
        results['general_evaluation'] = evaluate_lifd(
            trained_models=trained_models,
            test_data=test_data,
            config=vars(args)
        )
    
    if args.run_exp1:
        print("\n=== Running Experiment 1: Dual-Channel Fingerprint Robustness ===")
        results['exp1'] = dual_channel_fingerprint_experiment(
            lifd_model=trained_models['lifd_model'],
            extraction_net=trained_models['extraction_net'],
            test_data=test_data,
            fingerprints=trained_models['fingerprints'],
            config=vars(args)
        )
    
    if args.run_exp2:
        print("\n=== Running Experiment 2: Ablation Study on Adaptive Balancing ===")
        results['exp2'] = ablation_study_experiment(
            lifd_model=trained_models['lifd_model'],
            extraction_net=trained_models['extraction_net'],
            test_data=test_data,
            fingerprints=trained_models['fingerprints'],
            config=vars(args)
        )
    
    if args.run_exp3:
        print("\n=== Running Experiment 3: Latent Fingerprint Injection Analysis ===")
        results['exp3'] = latent_fingerprint_injection_analysis(
            lifd_model=trained_models['lifd_model'],
            test_data=test_data,
            fingerprints=trained_models['fingerprints'],
            config=vars(args)
        )
    
    return results

def save_models(args, trained_models):
    """Save the trained models."""
    print("\nSaving models...")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.save(
        trained_models['lifd_model'].state_dict(),
        os.path.join(args.save_dir, "lifd_model_final.pth")
    )
    
    torch.save(
        trained_models['extraction_net'].state_dict(),
        os.path.join(args.save_dir, "extraction_net_final.pth")
    )
    
    print(f"Models saved to {args.save_dir}")

def print_summary(args, results):
    """Print a summary of the experiment results."""
    print("\n" + "="*80)
    print("LIFD EXPERIMENT SUMMARY")
    print("="*80)
    
    print("\nGeneral Settings:")
    print(f"- Random Seed: {args.seed}")
    print(f"- Device: {args.device}")
    print(f"- Image Size: {args.image_size}")
    print(f"- Fingerprint Dimension: {args.fingerprint_dim}")
    print(f"- Number of Users: {args.num_users}")
    
    print("\nModel Settings:")
    print(f"- Mode: {args.mode} ({'Parameter-Only' if args.mode == 'A' else 'Latent-Only' if args.mode == 'B' else 'Dual-Channel'})")
    print(f"- Adaptive Alpha: {args.adaptive_alpha}")
    
    if 'general_evaluation' in results and 'general_metrics' in results['general_evaluation']:
        metrics = results['general_evaluation']['general_metrics']
        print("\nGeneral Evaluation Results:")
        print(f"- Clean Extraction Accuracy: {metrics['clean_extraction_acc']:.4f}")
        print(f"- Attacked (Blur+JPEG) Accuracy: {metrics['attacked_blur_jpeg_acc']:.4f}")
        print(f"- Attacked (Noise) Accuracy: {metrics['attacked_noise_acc']:.4f}")
        print(f"- MSE: {metrics['mse']:.4f}")
        print(f"- PSNR: {metrics['psnr']:.2f} dB")
    
    if 'exp1' in results:
        print("\nExperiment 1 Results (Dual-Channel Fingerprint Robustness):")
        for mode in ['A', 'B', 'C']:
            mode_name = 'Parameter-Only' if mode == 'A' else 'Latent-Only' if mode == 'B' else 'Dual-Channel'
            print(f"- Mode {mode} ({mode_name}):")
            print(f"  - Clean Accuracy: {results['exp1'][mode]['clean']:.4f}")
            print(f"  - Blur+JPEG Accuracy: {results['exp1'][mode]['attacked_blur_jpeg']:.4f}")
            print(f"  - Noise Accuracy: {results['exp1'][mode]['attacked_noise']:.4f}")
    
    if 'exp2' in results:
        print("\nExperiment 2 Results (Ablation Study on Adaptive Balancing):")
        alphas = results['exp2']['alphas']
        mse_scores = results['exp2']['mse_scores']
        psnr_scores = results['exp2']['psnr_scores']
        acc_scores = results['exp2']['fingerprint_accuracies']
        
        for i, alpha in enumerate(alphas):
            print(f"- Alpha = {alpha}:")
            print(f"  - MSE: {mse_scores[i]:.4f}")
            print(f"  - PSNR: {psnr_scores[i]:.2f} dB")
            print(f"  - Extraction Accuracy: {acc_scores[i]:.4f}")
    
    if 'exp3' in results and results['exp3'] is not None:
        print("\nExperiment 3 Results (Latent Fingerprint Injection Analysis):")
        print(f"- Total Variation: {results['exp3']['total_variation']:.6f}")
    
    print("\nAll experiments completed successfully")
    print("="*80)
    print(f"Plots and logs saved to {args.log_dir}")
    print(f"Models saved to {args.save_dir}")
    print("="*80)

def test_code():
    """
    Test function that runs a very short version of each experiment.
    This should finish quickly to check basic code execution.
    """
    print("\n=== Running Test Function ===")
    
    config = {
        'seed': 42,
        'batch_size': 2,
        'image_size': 32,
        'use_dummy': True,
        'data_dir': None,
        'fingerprint_dim': 16,
        'num_users': 2,
        'num_epochs': 1,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'adaptive_alpha': 0.5,
        'mode': 'C',
        'device': 'cpu',
        'save_dir': 'models',
        'log_dir': 'logs'
    }
    
    try:
        print("Running preprocessing test...")
        preprocessed_data = preprocess_data(config)
        print("Preprocessing OK.")
        
        print("Initializing models...")
        device = torch.device(config['device'])
        lifd_model = LIFDModel(
            fingerprint_dim=config['fingerprint_dim'],
            image_size=config['image_size'],
            adaptive_alpha=config['adaptive_alpha']
        ).to(device)
        
        extraction_net = FingerprintExtractionNet(
            fingerprint_dim=config['fingerprint_dim'],
            image_size=config['image_size']
        ).to(device)
        
        lifd_model.set_mode(config['mode'])
        print("Models initialized OK.")
        
        print("Testing forward pass...")
        batch = preprocessed_data['train_data'][0].to(device)
        fingerprints = preprocessed_data['fingerprints'].to(device)
        with torch.no_grad():
            output = lifd_model(batch, fingerprints[0:batch.size(0)])
        print(f"Forward pass OK. Output shape: {output.shape}")
        
        print("Testing fingerprint extraction...")
        with torch.no_grad():
            extracted = extraction_net(output)
        print(f"Extraction OK. Extracted shape: {extracted.shape}")
        
        print("Test function finished successfully.")
        return True
    except Exception as e:
        print(f"Test function encountered an error: {e}")
        return False

def main():
    """Main function to run the LIFD experiments."""
    args = parse_args()
    
    device = setup_environment(args)
    
    start_time = time.time()
    
    print("\nPreprocessing data...")
    preprocessed_data = preprocess_data(vars(args))
    
    trained_models = load_or_train_models(args, preprocessed_data)
    
    save_models(args, trained_models)
    
    results = run_experiments(
        args=args,
        trained_models=trained_models,
        test_data=preprocessed_data['val_data']  # Use validation data as test data
    )
    
    print_summary(args, results)
    
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    if test_code():
        main()
    else:
        print("Test failed. Please check the error messages above.")
