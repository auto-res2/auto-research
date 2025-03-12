"""Evaluation code for experiments."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.models import SimpleCNN, MNISTNet
from src.utils.utils import set_seed, get_device, log_experiment_results
from config.experiment_config import (
    RANDOM_SEED,
    DEVICE,
)


def evaluate_synthetic_results(synthetic_results):
    """Evaluate and display results from synthetic optimization benchmarks.
    
    Args:
        synthetic_results (dict): Results from synthetic optimization benchmarks
    """
    print("\n=== Synthetic Optimization Results ===")
    
    # Evaluate quadratic function results
    quadratic_results = synthetic_results["quadratic"]
    quadratic_losses = quadratic_results["losses"]
    quadratic_final_positions = quadratic_results["final_positions"]
    
    print("\nQuadratic Function Optimization:")
    print("Final Loss Values:")
    for name, losses in quadratic_losses.items():
        print(f"  {name}: {losses[-1]:.6f}")
    
    print("\nFinal Positions:")
    for name, pos in quadratic_final_positions.items():
        print(f"  {name}: {pos}")
    
    # Evaluate Rosenbrock function results
    rosenbrock_results = synthetic_results["rosenbrock"]
    rosenbrock_losses = rosenbrock_results["losses"]
    rosenbrock_final_positions = rosenbrock_results["final_positions"]
    
    print("\nRosenbrock Function Optimization:")
    print("Final Loss Values:")
    for name, losses in rosenbrock_losses.items():
        print(f"  {name}: {losses[-1]:.6f}")
    
    print("\nFinal Positions:")
    for name, pos in rosenbrock_final_positions.items():
        print(f"  {name}: {pos}")
    
    # Calculate convergence rates
    print("\nConvergence Analysis:")
    
    for function_name, results in [("Quadratic", quadratic_losses), ("Rosenbrock", rosenbrock_losses)]:
        print(f"\n{function_name} Function:")
        
        for name, losses in results.items():
            # Calculate average loss reduction per iteration over the last 20% of iterations
            start_idx = int(len(losses) * 0.8)
            if start_idx < len(losses) - 1:  # Ensure we have at least 2 points
                avg_reduction = (losses[start_idx] - losses[-1]) / (len(losses) - start_idx)
                print(f"  {name} avg loss reduction per iteration: {avg_reduction:.8f}")
            
            # Calculate iterations to reach 1% of initial loss
            threshold = 0.01 * losses[0]
            for i, loss in enumerate(losses):
                if loss <= threshold:
                    print(f"  {name} iterations to reach 1% of initial loss: {i}")
                    break
            else:
                print(f"  {name} did not reach 1% of initial loss")


def evaluate_cifar10_model(model_path, test_loader):
    """Evaluate a trained CIFAR-10 model.
    
    Args:
        model_path (str): Path to the saved model
        test_loader: DataLoader for test data
        
    Returns:
        dict: Evaluation results
    """
    print(f"\n=== Evaluating CIFAR-10 Model: {model_path} ===")
    set_seed(RANDOM_SEED)
    device = get_device()
    
    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Evaluate on test set
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # Calculate overall accuracy
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Calculate per-class accuracy
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nPer-class Accuracy:")
    for i in range(10):
        class_acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  {class_names[i]}: {class_acc:.2f}%")
    
    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    class_accuracies = [100.0 * class_correct[i] / class_total[i] for i in range(10)]
    plt.bar(class_names, class_accuracies)
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title(f"CIFAR-10 Per-class Accuracy - {model_path.split('/')[-1]}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"./logs/cifar10_class_acc_{model_path.split('/')[-1].split('.')[0]}.png")
    plt.close()
    
    return {
        "accuracy": accuracy,
        "per_class_accuracy": dict(zip(class_names, class_accuracies))
    }


def evaluate_mnist_model(model_path, test_loader):
    """Evaluate a trained MNIST model.
    
    Args:
        model_path (str): Path to the saved model
        test_loader: DataLoader for test data
        
    Returns:
        dict: Evaluation results
    """
    print(f"\n=== Evaluating MNIST Model: {model_path} ===")
    set_seed(RANDOM_SEED)
    device = get_device()
    
    # Load model
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Evaluate on test set
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # Calculate overall accuracy
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Calculate per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(10):
        class_acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  Digit {i}: {class_acc:.2f}%")
    
    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    class_accuracies = [100.0 * class_correct[i] / class_total[i] for i in range(10)]
    plt.bar([str(i) for i in range(10)], class_accuracies)
    plt.xlabel("Digit")
    plt.ylabel("Accuracy (%)")
    plt.title(f"MNIST Per-class Accuracy - {model_path.split('/')[-1]}")
    plt.tight_layout()
    plt.savefig(f"./logs/mnist_class_acc_{model_path.split('/')[-1].split('.')[0]}.png")
    plt.close()
    
    return {
        "accuracy": accuracy,
        "per_class_accuracy": dict(zip([str(i) for i in range(10)], class_accuracies))
    }


def evaluate_cifar10_results(cifar_results):
    """Evaluate and display results from CIFAR-10 training.
    
    Args:
        cifar_results (dict): Results from CIFAR-10 training
    """
    print("\n=== CIFAR-10 Training Results ===")
    
    # Compare final training loss
    print("\nFinal Training Loss:")
    for name, data in cifar_results.items():
        print(f"  {name}: {data['train_losses'][-1]:.4f}")
    
    # Compare final test accuracy
    print("\nFinal Test Accuracy:")
    for name, data in cifar_results.items():
        print(f"  {name}: {data['test_accuracies'][-1]:.2f}%")
    
    # Calculate convergence speed (epochs to reach 90% of final accuracy)
    print("\nConvergence Speed (epochs to reach 90% of final accuracy):")
    for name, data in cifar_results.items():
        final_acc = data['test_accuracies'][-1]
        threshold = 0.9 * final_acc
        
        for epoch, acc in enumerate(data['test_accuracies']):
            if acc >= threshold:
                print(f"  {name}: {epoch + 1}")
                break
        else:
            print(f"  {name}: Did not reach threshold")


def evaluate_mnist_ablation_results(mnist_results):
    """Evaluate and display results from MNIST ablation study.
    
    Args:
        mnist_results (dict): Results from MNIST ablation study
    """
    print("\n=== MNIST Ablation Study Results ===")
    
    # Compare final training loss
    print("\nFinal Training Loss:")
    for name, data in mnist_results.items():
        print(f"  {name}: {data['train_losses'][-1]:.4f}")
    
    # Compare final test accuracy
    print("\nFinal Test Accuracy:")
    for name, data in mnist_results.items():
        print(f"  {name}: {data['test_accuracies'][-1]:.2f}%")
    
    # Analyze effect of curvature influence parameter
    acm_results = {name: data for name, data in mnist_results.items() if name.startswith("ACM")}
    
    if acm_results:
        print("\nEffect of Curvature Influence Parameter:")
        
        # Extract curvature values and corresponding final accuracies
        curvature_values = []
        final_accuracies = []
        
        for name, data in acm_results.items():
            if name.startswith("ACM_curv_"):
                curv_value = float(name.split("_")[-1])
                final_acc = data['test_accuracies'][-1]
                
                curvature_values.append(curv_value)
                final_accuracies.append(final_acc)
        
        # Sort by curvature value
        sorted_indices = np.argsort(curvature_values)
        sorted_curvature = [curvature_values[i] for i in sorted_indices]
        sorted_accuracies = [final_accuracies[i] for i in sorted_indices]
        
        # Print sorted results
        for curv, acc in zip(sorted_curvature, sorted_accuracies):
            print(f"  Curvature influence = {curv}: {acc:.2f}%")
        
        # Plot effect of curvature influence
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_curvature, sorted_accuracies, 'o-')
        plt.xlabel("Curvature Influence Parameter")
        plt.ylabel("Final Test Accuracy (%)")
        plt.title("Effect of Curvature Influence Parameter on MNIST Accuracy")
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig("./logs/mnist_curvature_influence.png")
        plt.close()
