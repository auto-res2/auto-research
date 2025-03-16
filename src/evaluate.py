import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.optimizer import ACMOptimizer
from train import SimpleCNN, SimpleMLP

def plot_synthetic_results(results, save_dir='paper'):
    """
    Plot results from synthetic optimization experiments.
    
    Args:
        results (dict): Dictionary containing synthetic optimization results
        save_dir (str): Directory to save plots
    """
    print("Plotting synthetic optimization results...")
    
    # Create directory for plots
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot quadratic function optimization results
    plt.figure(figsize=(10, 6))
    for name, result in results['quadratic'].items():
        plt.plot(result['losses'], label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Quadratic Function Optimization')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f'{save_dir}/quadratic_optimization.png')
    plt.close()
    
    # Plot Rosenbrock function optimization results
    plt.figure(figsize=(10, 6))
    for name, result in results['rosenbrock'].items():
        plt.plot(result['losses'], label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Rosenbrock Function Optimization')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f'{save_dir}/rosenbrock_optimization.png')
    plt.close()
    
    # Print final results
    print("\nFinal Results for Quadratic Function Optimization:")
    for name, result in results['quadratic'].items():
        print(f"{name}: Final Loss = {result['final_loss']:.6f}, Final x = {result['final_x']}")
    
    print("\nFinal Results for Rosenbrock Function Optimization:")
    for name, result in results['rosenbrock'].items():
        print(f"{name}: Final Loss = {result['final_loss']:.6f}, Final x = {result['final_x']}")

def plot_cifar10_results(results, save_dir='paper'):
    """
    Plot results from CIFAR-10 training experiments.
    
    Args:
        results (dict): Dictionary containing CIFAR-10 training results
        save_dir (str): Directory to save plots
    """
    print("Plotting CIFAR-10 training results...")
    
    # Create directory for plots
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name)
    
    plt.xlabel('Iteration (x100 batches)')
    plt.ylabel('Loss')
    plt.title('CIFAR-10 Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/cifar10_training_loss.png')
    plt.close()
    
    # Plot test accuracies
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(range(1, len(result['test_accuracies']) + 1), result['test_accuracies'], label=name, marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CIFAR-10 Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/cifar10_test_accuracy.png')
    plt.close()
    
    # Print final results
    print("\nFinal Results for CIFAR-10 Training:")
    for name, result in results.items():
        print(f"{name}: Final Accuracy = {result['final_accuracy']:.2f}%, Training Time = {result['training_time']:.2f}s")

def plot_mnist_ablation_results(results, save_dir='paper'):
    """
    Plot results from MNIST ablation study.
    
    Args:
        results (dict): Dictionary containing MNIST ablation study results
        save_dir (str): Directory to save plots
    """
    print("Plotting MNIST ablation study results...")
    
    # Create directory for plots
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract hyperparameters and final accuracies
    lr_values = sorted(list(set([result['lr'] for result in results.values()])))
    beta_values = sorted(list(set([result['beta'] for result in results.values()])))
    curvature_values = sorted(list(set([result['curvature_influence'] for result in results.values()])))
    
    # Plot learning rate impact
    plt.figure(figsize=(10, 6))
    for beta in beta_values:
        for curv in curvature_values:
            accuracies = []
            for lr in lr_values:
                config_name = f"lr{lr}_beta{beta}_curv{curv}"
                if config_name in results:
                    accuracies.append(results[config_name]['final_accuracy'])
            
            if len(accuracies) == len(lr_values):
                plt.plot(lr_values, accuracies, marker='o', label=f'β={beta}, c={curv}')
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Accuracy (%)')
    plt.title('Impact of Learning Rate on MNIST Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.savefig(f'{save_dir}/mnist_lr_impact.png')
    plt.close()
    
    # Plot beta impact
    plt.figure(figsize=(10, 6))
    for lr in lr_values:
        for curv in curvature_values:
            accuracies = []
            for beta in beta_values:
                config_name = f"lr{lr}_beta{beta}_curv{curv}"
                if config_name in results:
                    accuracies.append(results[config_name]['final_accuracy'])
            
            if len(accuracies) == len(beta_values):
                plt.plot(beta_values, accuracies, marker='o', label=f'lr={lr}, c={curv}')
    
    plt.xlabel('Momentum Factor (β)')
    plt.ylabel('Final Accuracy (%)')
    plt.title('Impact of Momentum Factor on MNIST Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/mnist_beta_impact.png')
    plt.close()
    
    # Plot curvature influence impact
    plt.figure(figsize=(10, 6))
    for lr in lr_values:
        for beta in beta_values:
            accuracies = []
            for curv in curvature_values:
                config_name = f"lr{lr}_beta{beta}_curv{curv}"
                if config_name in results:
                    accuracies.append(results[config_name]['final_accuracy'])
            
            if len(accuracies) == len(curvature_values):
                plt.plot(curvature_values, accuracies, marker='o', label=f'lr={lr}, β={beta}')
    
    plt.xlabel('Curvature Influence')
    plt.ylabel('Final Accuracy (%)')
    plt.title('Impact of Curvature Influence on MNIST Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.savefig(f'{save_dir}/mnist_curvature_impact.png')
    plt.close()
    
    # Print best configuration
    best_config = max(results.items(), key=lambda x: x[1]['final_accuracy'])
    print("\nBest Configuration for MNIST:")
    print(f"Config: {best_config[0]}")
    print(f"Learning Rate: {best_config[1]['lr']}")
    print(f"Momentum Factor (β): {best_config[1]['beta']}")
    print(f"Curvature Influence: {best_config[1]['curvature_influence']}")
    print(f"Final Accuracy: {best_config[1]['final_accuracy']:.2f}%")
    print(f"Training Time: {best_config[1]['training_time']:.2f}s")

def evaluate_cifar10_models(test_loader, results, config=None):
    """
    Evaluate trained CIFAR-10 models.
    
    Args:
        test_loader: DataLoader for test data
        results (dict): Dictionary containing training results
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Updated results dictionary with evaluation metrics
    """
    print("\nEvaluating CIFAR-10 models...")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    for name in results.keys():
        print(f"\nEvaluating {name} model...")
        
        # Load model
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(f'models/cifar10/model_{name}.pt'))
        model.eval()
        
        # Evaluation metrics
        test_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        # Calculate metrics
        test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        # Update results
        results[name]['test_loss'] = test_loss
        results[name]['accuracy'] = accuracy
        
        # Per-class accuracy
        class_accuracy = {}
        for i in range(10):
            if class_total[i] > 0:
                class_accuracy[i] = 100 * class_correct[i] / class_total[i]
            else:
                class_accuracy[i] = 0
        
        results[name]['class_accuracy'] = class_accuracy
        
        # Print results
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("Per-class Accuracy:")
        for i in range(10):
            print(f"Class {i}: {class_accuracy[i]:.2f}%")
    
    return results

def evaluate_mnist_models(test_loader, results, config=None):
    """
    Evaluate trained MNIST models.
    
    Args:
        test_loader: DataLoader for test data
        results (dict): Dictionary containing training results
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Updated results dictionary with evaluation metrics
    """
    print("\nEvaluating MNIST models...")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    for config_name in results.keys():
        print(f"\nEvaluating model with config {config_name}...")
        
        # Load model
        model = SimpleMLP().to(device)
        model.load_state_dict(torch.load(f'models/mnist/model_{config_name}.pt'))
        model.eval()
        
        # Evaluation metrics
        test_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        # Calculate metrics
        test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        # Update results
        results[config_name]['test_loss'] = test_loss
        results[config_name]['accuracy'] = accuracy
        
        # Per-class accuracy
        class_accuracy = {}
        for i in range(10):
            if class_total[i] > 0:
                class_accuracy[i] = 100 * class_correct[i] / class_total[i]
            else:
                class_accuracy[i] = 0
        
        results[config_name]['class_accuracy'] = class_accuracy
        
        # Print results
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    return results

def evaluate_models(data, training_results, config=None):
    """
    Main function to evaluate all models.
    
    Args:
        data (dict): Dictionary containing all preprocessed data
        training_results (dict): Dictionary containing all training results
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Dictionary containing all evaluation results
    """
    # Plot synthetic optimization results
    plot_synthetic_results(training_results['synthetic'])
    
    # Evaluate and plot CIFAR-10 results
    cifar10_results = evaluate_cifar10_models(data['cifar10'][1], training_results['cifar10'], config)
    plot_cifar10_results(cifar10_results)
    
    # Evaluate and plot MNIST ablation study results
    mnist_results = evaluate_mnist_models(data['mnist'][1], training_results['mnist'], config)
    plot_mnist_ablation_results(mnist_results)
    
    # Return all evaluation results
    return {
        'synthetic': training_results['synthetic'],
        'cifar10': cifar10_results,
        'mnist': mnist_results
    }

if __name__ == "__main__":
    # When run directly, load trained models and evaluate
    from preprocess import preprocess_data
    import torch
    
    # Preprocess data
    data = preprocess_data()
    
    # Load training results
    try:
        synthetic_results = torch.load('models/synthetic/optimization_results.pt')
        cifar10_results = torch.load('models/cifar10/training_results.pt')
        mnist_results = torch.load('models/mnist/ablation_results.pt')
        
        training_results = {
            'synthetic': synthetic_results,
            'cifar10': cifar10_results,
            'mnist': mnist_results
        }
        
        # Evaluate models
        evaluate_models(data, training_results)
    except FileNotFoundError:
        print("Error: Trained model files not found. Please run train.py first.")
