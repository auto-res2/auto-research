import os
import yaml
import torch
import argparse
from datetime import datetime
from preprocess import get_dataset
from train import train_model
from evaluate import evaluate_model

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """
    Main function to run the entire experiment
    """
    parser = argparse.ArgumentParser(description='Run GSD experiments')
    parser.add_argument('--config', type=str, default='config/gsd_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--test_run', action='store_true',
                        help='Run a quick test with minimal iterations')
    parser.add_argument('--output_dir', type=str, default='models/output',
                        help='Directory to save output files')
    parser.add_argument('--checkpoint_path', type=str, default='models/gsd_model.pt',
                        help='Path to save model checkpoints')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and load model from checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation on a pretrained model')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.test_run:
        config['training']['test_run'] = True
        config['training']['epochs'] = 1
        config['evaluation']['num_samples'] = 16
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print experiment configuration
    print("\n=== Geometric Score Distillation (GSD) Experiment ===")
    print(f"Configuration: {args.config}")
    print(f"Test run: {args.test_run}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Skip training: {args.skip_training}")
    print(f"Evaluation only: {args.eval_only}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    if not args.eval_only:
        print("\n=== Loading Datasets ===")
        train_loader, val_loader = get_dataset(config)
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Train or load model
    if args.eval_only or args.skip_training:
        # Load pretrained model
        print("\n=== Loading Pretrained Model ===")
        from train import GSDModel
        model = GSDModel(config).to(device)
        
        try:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            print(f"Loaded model from {args.checkpoint_path}")
        except FileNotFoundError:
            print(f"Error: Checkpoint file {args.checkpoint_path} not found.")
            if args.eval_only:
                print("Cannot run evaluation without a pretrained model.")
                return
            else:
                print("Will train a new model instead.")
                args.skip_training = False
    
    if not args.eval_only and not args.skip_training:
        # Train model
        print("\n=== Training Model ===")
        checkpoint_path = os.path.join(run_dir, 'model.pt')
        model = train_model(config, train_loader, val_loader, checkpoint_path)
        print(f"Training completed. Model saved to {checkpoint_path}")
    
    # Evaluate model
    print("\n=== Evaluating Model ===")
    results = evaluate_model(model, config, output_dir=run_dir)
    
    # Save results
    results_path = os.path.join(run_dir, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    print(f"Evaluation results saved to {results_path}")
    
    print("\n=== Experiment Completed ===")
    print(f"All outputs saved to {run_dir}")

if __name__ == "__main__":
    main()
