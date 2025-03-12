import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.models import SimpleCNN, MNISTNet

def evaluate_synthetic_results(results_dict, experiment_name):
    """
    Evaluate and visualize results from synthetic optimization experiments
    """
    plt.figure(figsize=(10, 6))
    for name, losses in results_dict.items():
        plt.plot(losses, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{experiment_name} Optimization")
    plt.legend()
    
    # Save the figure
    os.makedirs('./logs', exist_ok=True)
    plt.savefig(f'./logs/{experiment_name.lower().replace(" ", "_")}_comparison.png')
    plt.close()
    
    # Print final losses
    print("\nFinal losses:")
    for name, losses in results_dict.items():
        print(f"{name}: {losses[-1]:.6f}")

def evaluate_cifar10(model_paths, testloader):
    """
    Evaluate trained models on CIFAR-10 test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    for model_path in model_paths:
        optimizer_name = os.path.basename(model_path).split('_')[1].split('.')[0]
        print(f"\nEvaluating {optimizer_name} model on CIFAR-10 test set")
        
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(testloader)
        test_acc = 100. * correct / total
        
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        results[optimizer_name] = {
            'test_loss': test_loss,
            'test_acc': test_acc
        }
    
    return results

def evaluate_mnist_ablation(results):
    """
    Evaluate and visualize results from MNIST ablation study
    """
    # Plot training loss curves
    plt.figure(figsize=(12, 8))
    for config_name, data in results.items():
        plt.plot(data['train_losses'], label=config_name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MNIST Ablation Study - Training Loss")
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('./logs', exist_ok=True)
    plt.savefig('./logs/mnist_ablation_train_loss.png')
    plt.close()
    
    # Plot test accuracy curves
    plt.figure(figsize=(12, 8))
    for config_name, data in results.items():
        plt.plot(data['test_accs'], label=config_name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("MNIST Ablation Study - Test Accuracy")
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('./logs/mnist_ablation_test_acc.png')
    plt.close()
    
    # Print final test accuracies
    print("\nFinal test accuracies:")
    for config_name, data in results.items():
        print(f"{config_name}: {data['final_test_acc']:.2f}%")
    
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]['final_test_acc'])
    print(f"\nBest configuration: {best_config[0]} with accuracy {best_config[1]['final_test_acc']:.2f}%")
    
    return best_config
