"""
Model evaluation module for ACM optimizer experiments.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm
import torchvision.models as models
from transformers import GPT2LMHeadModel

from src.utils.optimizers import ACM
from src.utils.experiment_utils import set_seed, get_device, load_model, ExperimentLogger
from src.preprocess import preprocess_data


def evaluate_synthetic_functions(results, config, logger):
    """
    Evaluate results from synthetic function optimization.
    
    Args:
        results (dict): Results dictionary from training
        config (dict): Configuration dictionary
        logger (ExperimentLogger): Logger for experiment results
        
    Returns:
        dict: Evaluation metrics
    """
    logger.log("Evaluating synthetic function optimization results...")
    
    # Metrics dictionary
    metrics = {}
    
    # Evaluate results for each function
    for func_name, func_results in results.items():
        logger.log(f"\nEvaluating {func_name} function results:")
        
        # Function-specific metrics
        func_metrics = {}
        
        # Evaluate each optimizer
        for opt_name, opt_results in func_results.items():
            # Get final function value
            final_f_val = opt_results["f_vals"][-1]
            
            # Get convergence rate (average decrease in function value per iteration)
            f_vals = opt_results["f_vals"]
            if len(f_vals) > 1:
                convergence_rate = (f_vals[0] - f_vals[-1]) / len(f_vals)
            else:
                convergence_rate = 0.0
            
            # Store metrics
            func_metrics[opt_name] = {
                "final_f_val": final_f_val,
                "convergence_rate": convergence_rate
            }
            
            # Log metrics
            logger.log(f"  {opt_name}:")
            logger.log(f"    Final function value: {final_f_val:.6f}")
            logger.log(f"    Convergence rate: {convergence_rate:.6f}")
        
        # Determine best optimizer for this function
        best_opt = min(func_metrics.items(), key=lambda x: x[1]["final_f_val"])[0]
        func_metrics["best_optimizer"] = best_opt
        
        logger.log(f"  Best optimizer for {func_name}: {best_opt}")
        
        # Store function metrics
        metrics[func_name] = func_metrics
    
    # Determine overall best optimizer
    opt_counts = {}
    for func_metrics in metrics.values():
        best_opt = func_metrics["best_optimizer"]
        opt_counts[best_opt] = opt_counts.get(best_opt, 0) + 1
    
    best_overall_opt = max(opt_counts.items(), key=lambda x: x[1])[0]
    metrics["best_overall_optimizer"] = best_overall_opt
    
    logger.log(f"\nBest overall optimizer: {best_overall_opt}")
    logger.log("Synthetic function evaluation completed.")
    
    return metrics


def evaluate_cifar10_resnet(model_path, test_loader, config, logger):
    """
    Evaluate ResNet-18 model on CIFAR-10 test set.
    
    Args:
        model_path (str): Path to saved model
        test_loader: DataLoader for test data
        config (dict): Configuration dictionary
        logger (ExperimentLogger): Logger for experiment results
        
    Returns:
        dict: Evaluation metrics
    """
    logger.log(f"Evaluating CIFAR-10 ResNet-18 model from {model_path}...")
    
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Get device
    device = get_device()
    
    # Create model
    model = models.resnet18(num_classes=config["model"]["num_classes"])
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0] * config["model"]["num_classes"]
    class_total = [0] * config["model"]["num_classes"]
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
            
            # If test_mode, only run a few batches
            if config["test_mode"] and batch_idx >= 2:
                break
    
    # Calculate metrics
    avg_test_loss = test_loss / (batch_idx + 1)
    accuracy = 100.0 * correct / total
    
    # Calculate per-class accuracy
    class_accuracy = [100.0 * c / max(1, t) for c, t in zip(class_correct, class_total)]
    
    # Log metrics
    logger.log(f"Test loss: {avg_test_loss:.4f}")
    logger.log(f"Test accuracy: {accuracy:.2f}%")
    logger.log("Per-class accuracy:")
    for i, acc in enumerate(class_accuracy):
        logger.log(f"  Class {i}: {acc:.2f}%")
    
    # Store metrics
    metrics = {
        "test_loss": avg_test_loss,
        "accuracy": accuracy,
        "class_accuracy": class_accuracy
    }
    
    logger.log("CIFAR-10 ResNet-18 evaluation completed.")
    return metrics


def evaluate_transformer_lm(model_path, dataloader, config, logger):
    """
    Evaluate transformer language model.
    
    Args:
        model_path (str): Path to saved model
        dataloader: DataLoader for evaluation data
        config (dict): Configuration dictionary
        logger (ExperimentLogger): Logger for experiment results
        
    Returns:
        dict: Evaluation metrics
    """
    logger.log(f"Evaluating transformer language model from {model_path}...")
    
    # Set random seed for reproducibility
    set_seed(config["random_seed"])
    
    # Get device
    device = get_device()
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(config["model"]["name"])
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Evaluation
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = batch.to(device)
            
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            # If test_mode, only run a few batches
            if config["test_mode"] and batch_idx >= 2:
                break
    
    # Calculate metrics
    avg_loss = total_loss / (batch_idx + 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Log metrics
    logger.log(f"Test loss: {avg_loss:.4f}")
    logger.log(f"Perplexity: {perplexity:.4f}")
    
    # Store metrics
    metrics = {
        "test_loss": avg_loss,
        "perplexity": perplexity
    }
    
    logger.log("Transformer language model evaluation completed.")
    return metrics


def evaluate_model(results_path, model_path, config_path, experiment_type):
    """
    Evaluate model based on experiment type.
    
    Args:
        results_path (str): Path to results file
        model_path (str): Path to saved model
        config_path (str): Path to configuration file
        experiment_type (str): Type of experiment ('synthetic', 'cifar10', or 'transformer')
        
    Returns:
        dict: Evaluation metrics
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create logger
    logger = ExperimentLogger(config["output_dir"], f"{config['experiment_name']}_eval")
    logger.log("Starting model evaluation...")
    
    # Evaluate model based on experiment type
    if experiment_type == 'synthetic':
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        metrics = evaluate_synthetic_functions(results, config, logger)
    
    elif experiment_type == 'cifar10':
        # Preprocess data to get test loader
        _, test_loader = preprocess_data(config_path, experiment_type)[0]
        
        metrics = evaluate_cifar10_resnet(model_path, test_loader, config, logger)
    
    elif experiment_type == 'transformer':
        # Preprocess data to get dataloader
        dataloader, _ = preprocess_data(config_path, experiment_type)[0]
        
        metrics = evaluate_transformer_lm(model_path, dataloader, config, logger)
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Save metrics
    metrics_path = os.path.join(config["output_dir"], f"{config['experiment_name']}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.log(f"Metrics saved to {metrics_path}")
    logger.log("Model evaluation completed.")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model evaluation for ACM optimizer experiments')
    parser.add_argument('--results', type=str, help='Path to results file (for synthetic experiments)')
    parser.add_argument('--model', type=str, help='Path to saved model (for cifar10 and transformer experiments)')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True, choices=['synthetic', 'cifar10', 'transformer'], help='Type of experiment')
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_model(args.results, args.model, args.config, args.experiment)
    
    print(f"Model evaluation for {args.experiment} experiment completed.")
