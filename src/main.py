"""Main script for running CSTD experiments."""

import torch
import os
import sys
import time

# Add the repository root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import cstd_config as cfg
from src.preprocess import load_data, set_seed
from src.train import train_denoiser, train_refiner, train_distilled_model
from src.evaluate import evaluate_detection, compare_models

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['logs', 'models', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Set up logging for the experiments."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Redirect stdout and stderr to files
    sys.stdout = open(os.path.join('logs', 'stdout.log'), 'w')
    sys.stderr = open(os.path.join('logs', 'stderr.log'), 'w')

def test_experiments(device):
    """Quick test run of all experiments to verify code execution."""
    print("\n=== Starting quick tests for CSTD experiments ===")
    
    # Set test mode to True for quick run
    cfg.TEST_MODE = True
    
    # Load a small subset of data
    train_loader, test_loader = load_data(
        cfg.DATASET, cfg.BATCH_SIZE, test_mode=True, test_subset_size=cfg.TEST_SUBSET_SIZE)
    
    # Run quick tests for each experiment
    denoiser, trigger, mask = train_denoiser(
        train_loader, device, epochs=1, sigma1=cfg.EXP1_SIGMA1, sigma2=cfg.EXP1_SIGMA2, 
        consistency_weight=cfg.EXP1_CONSISTENCY_WEIGHT)
    
    refiner, trigger, mask = train_refiner(
        train_loader, device, epochs=1, num_steps=cfg.EXP2_NUM_STEPS, 
        threshold=cfg.EXP2_CONVERGENCE_THRESHOLD)
    
    tpr, tnr = evaluate_detection(refiner, test_loader, trigger, mask, device)
    print(f"Detection metrics - TPR: {tpr:.3f}, TNR: {tnr:.3f}")
    
    full_model, distilled_model = train_distilled_model(train_loader, device, epochs=1)
    
    compare_metrics = compare_models(full_model, distilled_model, test_loader, device)
    
    print("\n=== All experiments completed quick test run ===")
    return denoiser, refiner, full_model, distilled_model

def run_full_experiments(device):
    """Run full experiments for CSTD."""
    print("\n=== Starting full experiments for CSTD ===")
    
    # Set test mode to False for full run
    cfg.TEST_MODE = False
    
    # Load full dataset
    train_loader, test_loader = load_data(cfg.DATASET, cfg.BATCH_SIZE)
    
    # Run Experiment 1: Ambient-Consistent Trigger Estimation
    denoiser, trigger, mask = train_denoiser(
        train_loader, device, epochs=cfg.EXP1_EPOCHS, sigma1=cfg.EXP1_SIGMA1, 
        sigma2=cfg.EXP1_SIGMA2, consistency_weight=cfg.EXP1_CONSISTENCY_WEIGHT)
    
    # Save the trained model
    torch.save(denoiser.state_dict(), os.path.join('models', 'denoiser.pth'))
    
    # Run Experiment 2: Sequential Score-Based Trigger Refinement
    refiner, trigger, mask = train_refiner(
        train_loader, device, epochs=cfg.EXP2_EPOCHS, num_steps=cfg.EXP2_NUM_STEPS, 
        threshold=cfg.EXP2_CONVERGENCE_THRESHOLD)
    
    # Evaluate trigger detection
    tpr, tnr = evaluate_detection(refiner, test_loader, trigger, mask, device)
    print(f"Detection metrics - TPR: {tpr:.3f}, TNR: {tnr:.3f}")
    
    # Save the trained model
    torch.save(refiner.state_dict(), os.path.join('models', 'refiner.pth'))
    
    # Run Experiment 3: Fast Defense Distillation
    full_model, distilled_model = train_distilled_model(
        train_loader, device, epochs=cfg.EXP3_EPOCHS)
    
    # Compare models
    compare_metrics = compare_models(full_model, distilled_model, test_loader, device)
    
    # Save the trained models
    torch.save(full_model.state_dict(), os.path.join('models', 'full_model.pth'))
    torch.save(distilled_model.state_dict(), os.path.join('models', 'distilled_model.pth'))
    
    print("\n=== All experiments completed full run ===")
    return denoiser, refiner, full_model, distilled_model

def main():
    """Main function to run CSTD experiments."""
    print("\n" + "="*80)
    print("CONSISTENT SEQUENTIAL TRIGGER DEFENSE (CSTD) EXPERIMENTS")
    print("="*80)
    
    # Display experiment configuration
    print("\n[CONFIGURATION]")
    print(f"- Random Seed: {cfg.RANDOM_SEED}")
    print(f"- Dataset: {cfg.DATASET}")
    print(f"- Batch Size: {cfg.BATCH_SIZE}")
    print(f"- Trigger Patch Size: {cfg.TRIGGER_PATCH_SIZE}")
    print(f"- Trigger Implant Ratio: {cfg.TRIGGER_IMPLANT_RATIO}")
    print(f"- Test Mode: {cfg.TEST_MODE}")
    
    # Create necessary directories
    print("\n[SETUP] Creating necessary directories...")
    create_directories()
    print("- Created directories: logs/, models/, data/")
    
    # Set random seed for reproducibility
    print("\n[SETUP] Setting random seed for reproducibility...")
    set_seed(cfg.RANDOM_SEED)
    print(f"- Random seed set to {cfg.RANDOM_SEED}")
    
    # Set device
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n[HARDWARE] Using device: {device}")
    
    # Check if CUDA is available
    if device.type == 'cuda':
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        print(f"- Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("- Running on CPU. Note: Training will be slower without GPU acceleration.")
    
    # Setup logging for the GitHub workflow
    if not sys.stdout.isatty():
        print("\n[LOGGING] Setting up file logging for non-interactive session...")
        setup_logging()
        print("- Logs will be saved to logs/stdout.log and logs/stderr.log")
    
    # Run experiments
    print("\n[EXPERIMENTS] Starting experiments...")
    start_time = time.time()
    
    if cfg.TEST_MODE:
        print(f"- Running in TEST MODE with subset size: {cfg.TEST_SUBSET_SIZE}")
        print("- Each experiment will run for 1 epoch with a single batch")
        models = test_experiments(device)
    else:
        print("- Running FULL experiments")
        print(f"- Experiment 1: {cfg.EXP1_EPOCHS} epochs")
        print(f"- Experiment 2: {cfg.EXP2_EPOCHS} epochs")
        print(f"- Experiment 3: {cfg.EXP3_EPOCHS} epochs")
        models = run_full_experiments(device)
    
    total_time = time.time() - start_time
    print(f"\n[SUMMARY] All experiments completed in {total_time:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()
