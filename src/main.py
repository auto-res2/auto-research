import torch
import torch.nn as nn
from preprocess import load_cifar10
from train import (
    SimpleConvNet, HybridOptimizer, AggMoOptimizer,
    MADGRADOptimizer, train_model
)
from evaluate import evaluate_model

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10()
    
    # Initialize model
    print("Initializing model...")
    model = SimpleConvNet()
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizers for comparison
    optimizers = {
        'HybridOptimizer': HybridOptimizer(model.parameters(), lr=0.001),
        'AggMo': AggMoOptimizer(model.parameters(), lr=0.001),
        'MADGRAD': MADGRADOptimizer(model.parameters(), lr=0.001),
        'SGD': torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
        'Adam': torch.optim.Adam(model.parameters(), lr=0.001)
    }
    
    # Train and evaluate with each optimizer
    results = {}
    for opt_name, optimizer in optimizers.items():
        print(f"\nTraining with {opt_name}...")
        model = SimpleConvNet()  # Reset model for fair comparison
        metrics = train_model(model, trainloader, testloader, optimizer, criterion, epochs=5)
        
        print(f"\nFinal metrics for {opt_name}:")
        print(f"Training loss progression: {metrics['train_losses']}")
        print(f"Test loss progression: {metrics['test_losses']}")
        print(f"Test accuracy progression: {metrics['test_accuracies']}")
        print(f"Convergence rates: {metrics['convergence_rate']}")
        
        results[opt_name] = {
            'final_accuracy': metrics['test_accuracies'][-1],
            'final_loss': metrics['test_losses'][-1],
            'convergence_rates': metrics['convergence_rate'],
            'generalization_gap': metrics['test_losses'][-1] - metrics['train_losses'][-1]
        }
    
    # Print final results
    print("\nFinal Results:")
    for opt_name, metrics in results.items():
        print(f"\n{opt_name}:")
        print(f"Final Test Accuracy: {metrics['final_accuracy']:.2f}%")
        print(f"Final Test Loss: {metrics['final_loss']:.3f}")
        print(f"Average Convergence Rate: {sum(metrics['convergence_rates'])/len(metrics['convergence_rates']):.6f}")
        print(f"Generalization Gap: {metrics['generalization_gap']:.3f}")

if __name__ == "__main__":
    main()
