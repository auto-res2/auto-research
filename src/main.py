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
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Load configuration
    with open('config/default.yaml') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    train_loader, test_loader = prepare_cifar10(
        config['data']['cifar10']['data_dir']
    )
    
    # Initialize model
    model = SimpleCNN()
    
    # Train and evaluate
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
        experiment_name='cifar10_hybrid'
    )
    
    # Final evaluation
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
    print("\nFinal Results:")
    print(f"Training Loss: {metrics['final_train_loss']:.4f}")
    print(f"Validation Loss: {metrics['final_val_loss']:.4f}")
    print(f"Validation Accuracy: {metrics['final_val_acc']:.2f}%")
    print(f"Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['test_acc']:.2f}%")

if __name__ == '__main__':
    main()
