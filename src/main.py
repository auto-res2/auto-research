"""
Main script for running ACM optimizer experiments.
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils.experiment_utils import set_seed, get_device, ExperimentLogger


def run_experiment(config_path, experiment_type, test_mode=False):
    """
    Run experiment based on configuration and experiment type.
    
    Args:
        config_path (str): Path to configuration file
        experiment_type (str): Type of experiment ('synthetic', 'cifar10', or 'transformer')
        test_mode (bool): Whether to run in test mode (quick run with minimal data)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override test_mode if specified
    if test_mode:
        config["test_mode"] = True
    
    # Create output directories
    os.makedirs(config["output_dir"], exist_ok=True)
    if experiment_type != 'synthetic' and config["training"]["save_model"]:
        os.makedirs(config["training"]["save_dir"], exist_ok=True)
    
    # Create logger
    logger = ExperimentLogger("logs", f"{experiment_type}_experiment")
    
    # Log experiment start
    logger.log("=" * 80)
    logger.log(f"Starting {experiment_type.upper()} experiment with ACM optimizer")
    logger.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Device: {get_device()}")
    logger.log(f"Test mode: {config['test_mode']}")
    logger.log("=" * 80)
    
    # Log configuration
    logger.log("\nConfiguration:")
    logger.log_config(config)
    
    # Step 1: Preprocess data
    logger.log("\n" + "=" * 40)
    logger.log("STEP 1: DATA PREPROCESSING")
    logger.log("=" * 40)
    
    data, _ = preprocess_data(config_path, experiment_type)
    
    # Step 2: Train model
    logger.log("\n" + "=" * 40)
    logger.log("STEP 2: MODEL TRAINING")
    logger.log("=" * 40)
    
    results, _ = train_model(config_path, experiment_type)
    
    # Save results
    results_path = os.path.join(config["output_dir"], f"{config['experiment_name']}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.log(f"\nResults saved to {results_path}")
    
    # Step 3: Evaluate model
    logger.log("\n" + "=" * 40)
    logger.log("STEP 3: MODEL EVALUATION")
    logger.log("=" * 40)
    
    if experiment_type == 'synthetic':
        metrics = evaluate_model(results_path, None, config_path, experiment_type)
    else:
        model_path = os.path.join(
            config["training"]["save_dir"],
            f"{'resnet18_ACM' if experiment_type == 'cifar10' else 'transformer_lm'}.pth"
        )
        metrics = evaluate_model(None, model_path, config_path, experiment_type)
    
    # Log summary
    logger.log("\n" + "=" * 40)
    logger.log("EXPERIMENT SUMMARY")
    logger.log("=" * 40)
    
    if experiment_type == 'synthetic':
        logger.log(f"Best overall optimizer: {metrics['best_overall_optimizer']}")
        
        for func_name, func_metrics in metrics.items():
            if func_name != 'best_overall_optimizer':
                logger.log(f"\n{func_name.capitalize()} function:")
                logger.log(f"  Best optimizer: {func_metrics['best_optimizer']}")
                
                for opt_name, opt_metrics in func_metrics.items():
                    if opt_name != 'best_optimizer':
                        logger.log(f"  {opt_name}:")
                        logger.log(f"    Final function value: {opt_metrics['final_f_val']:.6f}")
                        logger.log(f"    Convergence rate: {opt_metrics['convergence_rate']:.6f}")
    
    elif experiment_type == 'cifar10':
        logger.log(f"Test accuracy: {metrics['accuracy']:.2f}%")
        logger.log(f"Test loss: {metrics['test_loss']:.4f}")
        
        logger.log("\nPer-class accuracy:")
        for i, acc in enumerate(metrics['class_accuracy']):
            logger.log(f"  Class {i}: {acc:.2f}%")
        
        # Compare with baseline
        if 'Adam' in results:
            acm_acc = results['ACM']['final_val_accuracy']
            adam_acc = results['Adam']['final_val_accuracy']
            
            logger.log("\nComparison with baseline:")
            logger.log(f"  ACM validation accuracy: {acm_acc:.2f}%")
            logger.log(f"  Adam validation accuracy: {adam_acc:.2f}%")
            logger.log(f"  Improvement: {acm_acc - adam_acc:.2f}%")
    
    elif experiment_type == 'transformer':
        logger.log(f"Test perplexity: {metrics['perplexity']:.4f}")
        logger.log(f"Test loss: {metrics['test_loss']:.4f}")
    
    logger.log("\n" + "=" * 80)
    logger.log(f"{experiment_type.upper()} experiment completed successfully!")
    logger.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 80)


def run_all_experiments(test_mode=False):
    """
    Run all experiments (synthetic, CIFAR-10, and transformer).
    
    Args:
        test_mode (bool): Whether to run in test mode (quick run with minimal data)
    """
    print("=" * 80)
    print("RUNNING ALL EXPERIMENTS WITH ACM OPTIMIZER")
    print("=" * 80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run synthetic function experiments
    config_path = "config/synthetic_experiment_config.json"
    run_experiment(config_path, "synthetic", test_mode)
    
    # Run CIFAR-10 experiments
    config_path = "config/cifar10_experiment_config.json"
    run_experiment(config_path, "cifar10", test_mode)
    
    # Run transformer experiments
    config_path = "config/transformer_experiment_config.json"
    run_experiment(config_path, "transformer", test_mode)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ACM optimizer experiments')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--experiment', type=str, choices=['synthetic', 'cifar10', 'transformer', 'all'], default='all', help='Type of experiment')
    parser.add_argument('--test', action='store_true', help='Run in test mode (quick run with minimal data)')
    
    args = parser.parse_args()
    
    if args.experiment == 'all':
        run_all_experiments(args.test)
    else:
        if args.config is None:
            args.config = f"config/{args.experiment}_experiment_config.json"
        
        run_experiment(args.config, args.experiment, args.test)
