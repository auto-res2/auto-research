import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json

from preprocess import prepare_cifar10
from train import train_model
from evaluate import evaluate_model
from utils.models import SimpleCNN

def main():
    """Run the optimization experiment."""
    print("\n=== Starting Optimization Experiment ===\n")
    
    # Create necessary directories
    print("Setting up directories...")
    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    print("✓ Directories created successfully")
    
    # Load configuration
    print("\nLoading configuration...")
    with open('config/default.yaml') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded successfully")
    
    # Set device
    print("\nSetting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Prepare data
    print("\nPreparing CIFAR-10 dataset...")
    train_loader, test_loader = prepare_cifar10(
        config['data']['cifar10']['data_dir']
    )
    print("✓ Dataset prepared successfully")
    
    # Initialize model
    print("\nInitializing model...")
    model = SimpleCNN()
    print("✓ Model initialized successfully")
    
    # Train and evaluate
    print("\nStarting training...")
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
        experiment_name='cifar10_hybrid'
    )
    
    print("\nPerforming final evaluation...")
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Combine all metrics
    results = {
        'config': config,
        'training_metrics': metrics,
        'test_metrics': test_metrics
    }
    
    # Save results
    with open('logs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final metrics to stdout (will be captured in output.txt)
    print("\n=== Final Results ===")
    print("\nTraining Metrics:")
    print(f"  • Final Training Loss:    {metrics['final_train_loss']:.4f}")
    print(f"  • Final Validation Loss:  {metrics['final_val_loss']:.4f}")
    print(f"  • Best Validation Loss:   {metrics['best_val_loss']:.4f}")
    print(f"  • Validation Accuracy:    {metrics['final_val_acc']:.2f}%")
    
    print("\nTest Metrics:")
    print(f"  • Test Loss:              {test_metrics['test_loss']:.4f}")
    print(f"  • Test Accuracy:          {test_metrics['test_acc']:.2f}%")
    
    print("\nExperiment completed successfully! 🎉")
    print("Results have been saved to logs/results.json")

if __name__ == '__main__':
    main()
