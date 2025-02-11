import torch
import torch.nn as nn
from train import SimpleConvNet, train_model, HybridOptimizer
from evaluate import evaluate_model
from preprocess import get_dataloaders
import json
from pathlib import Path

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    trainloader, testloader = get_dataloaders(batch_size=128)
    
    # Initialize model
    model = SimpleConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizers for comparison
    optimizers = {
        'hybrid': HybridOptimizer(model.parameters(), lr=0.001),
        'sgd': torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
        'adam': torch.optim.Adam(model.parameters(), lr=0.001)
    }
    
    results = {}
    
    # Train and evaluate with each optimizer
    for opt_name, optimizer in optimizers.items():
        print(f"\nTraining with {opt_name} optimizer...")
        
        # Reset model weights
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
        # Train
        train_history = train_model(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=5  # Reduced epochs for testing
        )
        
        # Evaluate
        eval_results = evaluate_model(
            model=model,
            testloader=testloader,
            criterion=criterion,
            device=device
        )
        
        results[opt_name] = {
            'training_history': train_history,
            'evaluation_results': eval_results
        }
    
    # Save results
    output_dir = Path('logs')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nExperiment completed. Results saved to logs/experiment_results.json")

if __name__ == "__main__":
    main()
