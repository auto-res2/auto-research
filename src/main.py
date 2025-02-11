import torch
import torch.nn as nn
from train import SimpleCNN, train_cifar10
from evaluate import evaluate_cifar10
from preprocess import get_cifar10_loaders
import json
from datetime import datetime
import os

def run_cifar10_experiment(device: torch.device,
                          epochs: int = 10,
                          batch_size: int = 128):
    """Run CIFAR-10 experiments with different optimizers."""
    print("\n=== Starting CIFAR-10 Experiments ===")
    
    # Get data
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    
    # Optimizer configurations
    optimizer_configs = {
        'SGD': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
        'Adam': {'lr': 0.001, 'weight_decay': 5e-4},
        'MADGRAD': {'lr': 0.001, 'weight_decay': 5e-4},
        'Hybrid': {'lr': 0.001, 'weight_decay': 5e-4, 'eps': 1e-8}
    }
    
    results = {}
    
    for opt_name, opt_config in optimizer_configs.items():
        print(f"\nTraining with {opt_name}")
        model = SimpleCNN()
        
        # Training
        train_metrics = train_cifar10(
            model=model,
            train_loader=train_loader,
            optimizer_name=opt_name.lower(),
            device=device,
            epochs=epochs,
            **opt_config
        )
        
        # Evaluation
        eval_metrics = evaluate_cifar10(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        results[opt_name] = {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }
        
        print(f"{opt_name} Results:")
        print(f"Final Train Loss: {train_metrics['train_loss'][-1]:.4f}")
        print(f"Final Train Accuracy: {train_metrics['train_acc'][-1]:.2f}%")
        print(f"Test Loss: {eval_metrics['test_loss']:.4f}")
        print(f"Test Accuracy: {eval_metrics['test_accuracy']:.2f}%")
    
    return results

# PTB experiment removed for minimal testing
# Will be re-added when torchtext issues are resolved

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run only CIFAR-10 experiment with minimal epochs for testing
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'cifar10': run_cifar10_experiment(device, epochs=1, batch_size=32)  # Reduced epochs and batch size for quick testing
    }
    
    # Save results
    os.makedirs('logs', exist_ok=True)
    with open(f'logs/experiment_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
