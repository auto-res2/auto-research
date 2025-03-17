"""
Model evaluation module for ACM optimizer experiments.

This module implements evaluation functions and visualization for:
1. ResNet-18 on CIFAR-10 (Real-World Convergence Experiment)
2. Optimization on synthetic functions (Synthetic Loss Landscape Experiment)
3. Simple CNN with hyperparameter grid search (Hyperparameter Sensitivity Experiment)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import time

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_real_world_experiment(results):
    """
    Evaluate and visualize results from the real-world convergence experiment.
    
    Args:
        results (dict): Dictionary containing results for each optimizer
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n=== Evaluating Real-World Convergence Experiment ===")
    
    # Create directory for saving plots
    os.makedirs("logs/plots", exist_ok=True)
    
    # Extract metrics for each optimizer
    metrics = ["Train Loss", "Val Loss", "Val Accuracy", "Epoch Time"]
    metric_keys = ["train_loss", "val_loss", "val_acc", "epoch_time"]
    
    # Create figure for learning curves
    plt.figure(figsize=(16, 12))
    
    # Plot each metric
    for i, (metric, key) in enumerate(zip(metrics, metric_keys)):
        plt.subplot(2, 2, i+1)
        
        for opt_name in results.keys():
            values = results[opt_name][key]
            plt.plot(values, label=opt_name, marker='o')
        
        plt.title(metric)
        plt.xlabel("Epoch")
        if key == "val_acc":
            plt.ylabel("Accuracy (%)")
        else:
            plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("logs/plots/real_world_experiment.png")
    print(f"Saved plot to logs/plots/real_world_experiment.png")
    plt.close()
    
    # Calculate final metrics
    final_metrics = {}
    for opt_name in results.keys():
        final_metrics[opt_name] = {
            "final_train_loss": results[opt_name]["train_loss"][-1],
            "final_val_loss": results[opt_name]["val_loss"][-1],
            "final_val_acc": results[opt_name]["val_acc"][-1],
            "avg_epoch_time": np.mean(results[opt_name]["epoch_time"]),
            "total_time": np.sum(results[opt_name]["epoch_time"])
        }
    
    # Print final metrics
    print("\nFinal Metrics:")
    for opt_name, metrics in final_metrics.items():
        print(f"\n{opt_name} Optimizer:")
        print(f"  Final Training Loss: {metrics['final_train_loss']:.4f}")
        print(f"  Final Validation Loss: {metrics['final_val_loss']:.4f}")
        print(f"  Final Validation Accuracy: {metrics['final_val_acc']:.2f}%")
        print(f"  Average Epoch Time: {metrics['avg_epoch_time']:.2f}s")
        print(f"  Total Training Time: {metrics['total_time']:.2f}s")
    
    return final_metrics

def evaluate_synthetic_experiment(results):
    """
    Evaluate and visualize results from the synthetic loss landscape experiment.
    
    Args:
        results (dict): Dictionary containing results for each function and optimizer
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n=== Evaluating Synthetic Loss Landscape Experiment ===")
    
    # Create directory for saving plots
    os.makedirs("logs/plots", exist_ok=True)
    
    # Extract results
    quadratic_results = results["quadratic"]
    rosenbrock_results = results["rosenbrock"]
    
    # Evaluate quadratic function optimization
    plt.figure(figsize=(12, 5))
    
    # Plot trajectories
    plt.subplot(1, 2, 1)
    plt.plot(
        quadratic_results["acm_trajectory"][:, 0],
        quadratic_results["acm_trajectory"][:, 1],
        'o-',
        label='ACM'
    )
    plt.plot(
        quadratic_results["sgd_trajectory"][:, 0],
        quadratic_results["sgd_trajectory"][:, 1],
        's--',
        label='SGD'
    )
    plt.title("Quadratic Function Trajectories")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot adaptive learning rates
    plt.subplot(1, 2, 2)
    plt.plot(quadratic_results["acm_lr_evolution"], 'o-', color="purple")
    plt.title("ACM Adaptive LR Evolution (Quadratic)")
    plt.xlabel("Iteration")
    plt.ylabel("Adaptive LR")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("logs/plots/synthetic_quadratic.png")
    print(f"Saved plot to logs/plots/synthetic_quadratic.png")
    plt.close()
    
    # Evaluate Rosenbrock function optimization
    plt.figure(figsize=(12, 5))
    
    # Plot trajectories
    plt.subplot(1, 2, 1)
    plt.plot(
        rosenbrock_results["acm_trajectory"][:, 0],
        rosenbrock_results["acm_trajectory"][:, 1],
        'o-',
        label='ACM'
    )
    plt.plot(
        rosenbrock_results["sgd_trajectory"][:, 0],
        rosenbrock_results["sgd_trajectory"][:, 1],
        's--',
        label='SGD'
    )
    plt.title("Rosenbrock Function Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot adaptive learning rates
    plt.subplot(1, 2, 2)
    plt.plot(rosenbrock_results["acm_lr_evolution"], 'o-', color="green")
    plt.title("ACM Adaptive LR Evolution (Rosenbrock)")
    plt.xlabel("Iteration")
    plt.ylabel("Adaptive LR")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("logs/plots/synthetic_rosenbrock.png")
    print(f"Saved plot to logs/plots/synthetic_rosenbrock.png")
    plt.close()
    
    # Calculate convergence metrics
    metrics = {
        "quadratic": {
            "acm_final_point": quadratic_results["acm_trajectory"][-1],
            "sgd_final_point": quadratic_results["sgd_trajectory"][-1],
            "acm_distance_to_origin": np.linalg.norm(quadratic_results["acm_trajectory"][-1]),
            "sgd_distance_to_origin": np.linalg.norm(quadratic_results["sgd_trajectory"][-1]),
            "acm_path_length": calculate_path_length(quadratic_results["acm_trajectory"]),
            "sgd_path_length": calculate_path_length(quadratic_results["sgd_trajectory"])
        },
        "rosenbrock": {
            "acm_final_point": rosenbrock_results["acm_trajectory"][-1],
            "sgd_final_point": rosenbrock_results["sgd_trajectory"][-1],
            "acm_distance_to_optimum": np.linalg.norm(rosenbrock_results["acm_trajectory"][-1] - np.array([1.0, 1.0])),
            "sgd_distance_to_optimum": np.linalg.norm(rosenbrock_results["sgd_trajectory"][-1] - np.array([1.0, 1.0])),
            "acm_path_length": calculate_path_length(rosenbrock_results["acm_trajectory"]),
            "sgd_path_length": calculate_path_length(rosenbrock_results["sgd_trajectory"])
        }
    }
    
    # Print metrics
    print("\nQuadratic Function Optimization:")
    print(f"  ACM final point: {metrics['quadratic']['acm_final_point']}")
    print(f"  SGD final point: {metrics['quadratic']['sgd_final_point']}")
    print(f"  ACM distance to origin: {metrics['quadratic']['acm_distance_to_origin']:.6f}")
    print(f"  SGD distance to origin: {metrics['quadratic']['sgd_distance_to_origin']:.6f}")
    print(f"  ACM path length: {metrics['quadratic']['acm_path_length']:.6f}")
    print(f"  SGD path length: {metrics['quadratic']['sgd_path_length']:.6f}")
    
    print("\nRosenbrock Function Optimization:")
    print(f"  ACM final point: {metrics['rosenbrock']['acm_final_point']}")
    print(f"  SGD final point: {metrics['rosenbrock']['sgd_final_point']}")
    print(f"  ACM distance to optimum (1,1): {metrics['rosenbrock']['acm_distance_to_optimum']:.6f}")
    print(f"  SGD distance to optimum (1,1): {metrics['rosenbrock']['sgd_distance_to_optimum']:.6f}")
    print(f"  ACM path length: {metrics['rosenbrock']['acm_path_length']:.6f}")
    print(f"  SGD path length: {metrics['rosenbrock']['sgd_path_length']:.6f}")
    
    return metrics

def evaluate_hyperparameter_experiment(results):
    """
    Evaluate and visualize results from the hyperparameter sensitivity experiment.
    
    Args:
        results (dict): Dictionary containing results for ACM and Adam
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n=== Evaluating Hyperparameter Sensitivity Experiment ===")
    
    # Create directory for saving plots
    os.makedirs("logs/plots", exist_ok=True)
    
    # Extract results
    acm_results = results["ACM"]
    adam_results = results["Adam"]
    
    # Convert to pandas DataFrames
    df_acm = pd.DataFrame(acm_results)
    df_adam = pd.DataFrame(adam_results)
    
    # Create heatmap for ACM hyperparameters
    plt.figure(figsize=(10, 8))
    # Create pivot table manually since we're getting type errors
    lr_values = sorted(df_acm['lr'].unique())
    beta_values = sorted(df_acm['beta'].unique())
    pivot_data = np.zeros((len(lr_values), len(beta_values)))
    
    for i, lr in enumerate(lr_values):
        for j, beta in enumerate(beta_values):
            mask = (df_acm['lr'] == lr) & (df_acm['beta'] == beta)
            if mask.any():
                # Get the first value directly
                filtered_df = df_acm[mask]
                if len(filtered_df) > 0:
                    pivot_data[i, j] = filtered_df['val_acc'].iloc[0]
                else:
                    pivot_data[i, j] = 0
    
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=beta_values, yticklabels=lr_values)
    plt.title("ACM Hyperparameter Sensitivity (Validation Accuracy)")
    plt.savefig("logs/plots/acm_hyperparameter_heatmap.png")
    print(f"Saved plot to logs/plots/acm_hyperparameter_heatmap.png")
    plt.close()
    
    # Create line plot for Adam learning rates
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_adam, x='lr', y='val_acc', marker="o", sort=False)
    plt.title("Adam Hyperparameter Sensitivity (Validation Accuracy)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("logs/plots/adam_hyperparameter_lineplot.png")
    print(f"Saved plot to logs/plots/adam_hyperparameter_lineplot.png")
    plt.close()
    
    # Find best hyperparameters
    if len(df_acm) > 0:
        # Find the row with maximum validation accuracy
        max_acc_row = None
        max_acc = -1
        for _, row in df_acm.iterrows():
            if row['val_acc'] > max_acc:
                max_acc = row['val_acc']
                max_acc_row = row
        best_acm = {col: max_acc_row[col] for col in df_acm.columns} if max_acc_row is not None else {}
    else:
        best_acm = {}
        
    if len(df_adam) > 0:
        # Find the row with maximum validation accuracy
        max_acc_row = None
        max_acc = -1
        for _, row in df_adam.iterrows():
            if row['val_acc'] > max_acc:
                max_acc = row['val_acc']
                max_acc_row = row
        best_adam = {col: max_acc_row[col] for col in df_adam.columns} if max_acc_row is not None else {}
    else:
        best_adam = {}
    
    # Print best hyperparameters
    print("\nBest Hyperparameters:")
    print(f"\nACM Optimizer:")
    print(f"  Learning Rate: {best_acm['lr']}")
    print(f"  Beta: {best_acm['beta']}")
    print(f"  Validation Accuracy: {best_acm['val_acc']:.2f}%")
    print(f"  Validation Loss: {best_acm['val_loss']:.4f}")
    
    print(f"\nAdam Optimizer:")
    print(f"  Learning Rate: {best_adam['lr']}")
    print(f"  Validation Accuracy: {best_adam['val_acc']:.2f}%")
    print(f"  Validation Loss: {best_adam['val_loss']:.4f}")
    
    # Calculate robustness metrics (standard deviation of accuracy across hyperparameters)
    acm_robustness = df_acm['val_acc'].std()
    adam_robustness = df_adam['val_acc'].std()
    
    print("\nRobustness Metrics (lower std dev = more robust):")
    print(f"  ACM Accuracy Std Dev: {acm_robustness:.2f}")
    print(f"  Adam Accuracy Std Dev: {adam_robustness:.2f}")
    
    # Return metrics
    metrics = {
        "best_acm": best_acm,
        "best_adam": best_adam,
        "acm_robustness": acm_robustness,
        "adam_robustness": adam_robustness
    }
    
    return metrics

def calculate_path_length(trajectory):
    """
    Calculate the total path length of an optimization trajectory.
    
    Args:
        trajectory (numpy.ndarray): Array of points in the trajectory
    
    Returns:
        float: Total path length
    """
    path_length = 0.0
    for i in range(1, len(trajectory)):
        path_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
    return path_length

def evaluate_model(model, test_loader):
    """
    Evaluate a model on the test set.
    
    Args:
        model: PyTorch model
        test_loader: PyTorch DataLoader for test data
    
    Returns:
        tuple: (val_loss, val_acc) - Validation loss and accuracy
    """
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate validation statistics
    val_loss = val_loss / len(test_loader.dataset)
    val_acc = 100.0 * correct / total
    eval_time = time.time() - start_time
    
    print(f"Evaluation completed in {eval_time:.2f}s")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    
    return val_loss, val_acc

if __name__ == "__main__":
    # Test path length calculation
    trajectory = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    path_length = calculate_path_length(trajectory)
    print(f"Test path length: {path_length}")  # Should be 4.0
