import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
from pathlib import Path

# Import components
from preprocess import get_dataloader
from train import ATBFNPipeline, train_model, FixedStepSDE, AdaptiveSDE
from evaluate import run_full_evaluation, quick_test_evaluation

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['config', 'data', 'logs', 'models']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def print_system_info():
    """Print system and GPU information"""
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("==========================\n")

def run_experiment_adaptive_tokenization(config, device):
    """
    Run Experiment 1: Adaptive Tokenization & Resolution Generalization
    """
    print("\n=== Experiment 1: Adaptive Tokenization & Resolution Generalization ===")
    
    # Create dataloaders for different resolutions
    dataloaders = {}
    for resolution in config['image_resolutions']:
        dataloader = get_dataloader(config, resolution=resolution)
        dataloaders[resolution] = dataloader
    
    # Get a batch of images from each resolution
    for resolution, loader in dataloaders.items():
        print(f"\nProcessing images at resolution: {resolution}x{resolution}")
        imgs, _ = next(iter(loader))
        print(f"Loaded batch of images with shape: {imgs.shape}")
        
        # Create tokenizer
        tokenizer = ATBFNPipeline(config).tokenizer
        tokens, complexity = tokenizer(imgs.to(device))
        
        print(f"Dynamic Tokenizer Output -- Tokens shape: {tokens.shape}")
        
        # Fixed grid baseline
        baseline_tokens = F.adaptive_avg_pool2d(imgs, (8, 8))
        B, C, H_tok, W_tok = baseline_tokens.shape
        baseline_tokens_flat = baseline_tokens.view(B, C, H_tok*W_tok).permute(0, 2, 1)
        
        print(f"Baseline (fixed grid) Tokens shape: {baseline_tokens_flat.shape}")
        print(f"Complexity map statistics: min={complexity.min().item():.4f}, "
              f"max={complexity.max().item():.4f}, mean={complexity.mean().item():.4f}")

def run_experiment_token_evolution(config, device):
    """
    Run Experiment 2: Token-based SDE Evolution with Integrated Noise Control
    """
    print("\n=== Experiment 2: Token-based SDE Evolution with Integrated Noise Control ===")
    
    # Parameters
    B, N, token_dim = 4, 64, 64  # Small batch for testing
    
    # Initialize random tokens
    initial_tokens = torch.randn(B, N, token_dim, device=device)
    print(f"Initial tokens shape: {initial_tokens.shape}")
    
    # Fixed-step evolution
    fixed_sde = FixedStepSDE(token_dim, dt=0.1).to(device)
    tokens_fixed = initial_tokens.clone()
    
    start_time = time.time()
    for i in range(5):  # 5 iterations
        tokens_fixed = fixed_sde(tokens_fixed)
    elapsed_fixed = time.time() - start_time
    
    print(f"Fixed-step SDE evolution completed in {elapsed_fixed:.4f} seconds.")
    print(f"Final token shape: {tokens_fixed.shape}")

    # Adaptive SDE evolution
    adaptive_sde = AdaptiveSDE(token_dim, base_dt=0.1).to(device)
    tokens_adaptive = initial_tokens.clone()
    
    start_time = time.time()
    for i in range(5):  # 5 iterations
        tokens_adaptive, uncertainty = adaptive_sde(tokens_adaptive)
        avg_unc = uncertainty.mean().item()
        print(f"Iteration {i+1} adaptive evolution -- avg uncertainty: {avg_unc:.4f}")
    elapsed_adaptive = time.time() - start_time
    
    print(f"Adaptive SDE evolution completed in {elapsed_adaptive:.4f} seconds.")
    print(f"Final token shape: {tokens_adaptive.shape}")

def run_experiment_cross_token_attention(config, device):
    """
    Run Experiment 3: Cross-token Attention for Global Coherence
    """
    print("\n=== Experiment 3: Cross-token Attention for Global Coherence ===")
    
    # Parameters
    B, N, token_dim = 8, 64, 64
    
    # Initialize random tokens
    initial_tokens = torch.randn(B, N, token_dim, device=device)
    
    # Run pipeline with cross-token attention
    print("Running ATBFN pipeline WITH cross-token attention...")
    model_with_attn = ATBFNPipeline(config, token_dim=token_dim, use_attention=True).to(device)
    
    with torch.no_grad():
        output_with = model_with_attn(initial_tokens)
    
    print(f"Output with attention has shape: {output_with['reconstructed'].shape}")
    
    # Run pipeline without cross-token attention
    print("\nRunning ATBFN pipeline WITHOUT cross-token attention...")
    model_without_attn = ATBFNPipeline(config, token_dim=token_dim, use_attention=False).to(device)
    
    with torch.no_grad():
        output_without = model_without_attn(initial_tokens)
    
    print(f"Output without attention has shape: {output_without['reconstructed'].shape}")
    
    # Calculate difference
    norm_diff = torch.norm(output_with['reconstructed'] - output_without['reconstructed']).item()
    print(f"Difference (norm) between outputs: {norm_diff:.4f}")

def train_and_evaluate(config, device):
    """
    Train and evaluate the AT-BFN model
    """
    print("\n=== Training and Evaluating AT-BFN Model ===")
    
    # Create dataloaders
    train_loader = get_dataloader(config, train=True)
    test_loader = get_dataloader(config, train=False)
    
    # Create model
    model = ATBFNPipeline(config).to(device)
    
    # Train model
    losses = train_model(model, train_loader, config, device)
    
    # Save model
    model_path = os.path.join('models', 'atbfn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Quick evaluation
    mse = quick_test_evaluation(model, test_loader, config, device)
    
    return model, mse

def test_all(config, device):
    """
    Quick test to verify that code executes correctly.
    Each experiment is run with one quick iteration or a single batch.
    """
    print("\n=== Starting quick tests for all experiments ===")
    
    # Experiment 1: Just one batch per resolution
    run_experiment_adaptive_tokenization(config, device)
    
    # Experiment 2: Quick token evolution experiment
    run_experiment_token_evolution(config, device)
    
    # Experiment 3: Run ablation comparison of attention module in ATBFNPipeline
    run_experiment_cross_token_attention(config, device)
    
    print("\nAll quick tests finished successfully.")

def main():
    """Main function to run the AT-BFN experiments"""
    # Setup directories
    setup_directories()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run AT-BFN experiments')
    parser.add_argument('--config', type=str, default='config/atbfn_config.yaml', help='Path to config file')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test only')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cuda or cpu)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override quick test setting if specified
    if args.quick_test:
        config['quick_test'] = True
    
    # Set device
    if args.device:
        config['device'] = args.device
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    
    # Print system information
    print_system_info()
    
    # Set random seed for reproducibility
    set_seed(config['random_seed'])
    
    print(f"Running on device: {device}")
    print(f"Quick test mode: {'enabled' if config.get('quick_test', False) else 'disabled'}")
    
    if config.get('quick_test', False):
        # Run quick tests
        test_all(config, device)
    else:
        # Run full experiments
        # Experiment 1: Adaptive Tokenization & Resolution Generalization
        run_experiment_adaptive_tokenization(config, device)
        
        # Experiment 2: Token-based SDE Evolution with Integrated Noise Control
        run_experiment_token_evolution(config, device)
        
        # Experiment 3: Cross-token Attention for Global Coherence
        run_experiment_cross_token_attention(config, device)
        
        # Train and evaluate model
        model, _ = train_and_evaluate(config, device)
        
        # Create dataloaders for different resolutions for full evaluation
        dataloaders = {}
        for resolution in config['image_resolutions']:
            dataloaders[resolution] = get_dataloader(config, resolution=resolution, train=False)
        
        # Run full evaluation
        results = run_full_evaluation(model, dataloaders, config, device)
        print("\n=== Full Evaluation Results ===")
        print(results)

if __name__ == '__main__':
    main()
