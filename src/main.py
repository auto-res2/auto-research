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

try:
    # Try relative imports first (when running as a module)
    from .preprocess import preprocess_data
    from .train import train_model
    from .evaluate import evaluate_model
    from .utils.experiment_utils import set_seed, get_device, ExperimentLogger
except ImportError:
    # Fall back to absolute imports (when running as a script)
    from preprocess import preprocess_data
    from train import train_model
    from evaluate import evaluate_model
    from utils.experiment_utils import set_seed, get_device, ExperimentLogger


def run_experiment(config_path, experiment_type, test_mode=False):
    """
    Run experiment based on configuration and experiment type.
    
    Args:
        config_path (str): Path to configuration file
        experiment_type (str): Type of experiment ('synthetic', 'cifar10', or 'transformer')
        test_mode (bool): Whether to run in test mode (quick run with minimal data)
    """
    # Print detailed experiment information to standard output
    print("\n" + "=" * 80)
    print(f"RUNNING {experiment_type.upper()} EXPERIMENT WITH ACM OPTIMIZER")
    print("=" * 80)
    print(f"Configuration file: {config_path}")
    print(f"Experiment type: {experiment_type}")
    print(f"Test mode: {test_mode}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
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
    
    # Print detailed results to standard output
    print("\n" + "=" * 60)
    print(f"RESULTS FOR {experiment_type.upper()} EXPERIMENT")
    print("=" * 60)
    
    if experiment_type == 'synthetic':
        for func_name, func_results in results.items():
            print(f"\n{func_name.capitalize()} function:")
            for opt_name, opt_results in func_results.items():
                print(f"  {opt_name} optimizer:")
                print(f"    Final function value: {opt_results.get('final_f_val', opt_results.get('final_value', 0.0)):.6f}")
                print(f"    Iterations: {opt_results.get('iterations', 0)}")
                print(f"    Convergence rate: {opt_results.get('convergence_rate', 'N/A')}")
    
    elif experiment_type == 'cifar10':
        for opt_name, opt_results in results.items():
            print(f"\n{opt_name} optimizer:")
            print(f"  Final training accuracy: {opt_results.get('final_train_accuracy', 0.0):.2f}%")
            print(f"  Final validation accuracy: {opt_results.get('final_val_accuracy', 0.0):.2f}%")
            print(f"  Final training loss: {opt_results.get('final_train_loss', 0.0):.4f}")
            print(f"  Final validation loss: {opt_results.get('final_val_loss', 0.0):.4f}")
    
    elif experiment_type == 'transformer':
        print(f"\nFinal training loss: {results.get('final_train_loss', 0.0):.4f}")
        print(f"Final training perplexity: {results.get('final_train_perplexity', 0.0):.4f}")
        if 'final_val_loss' in results:
            print(f"Final validation loss: {results.get('final_val_loss', 0.0):.4f}")
            print(f"Final validation perplexity: {results.get('final_val_perplexity', 0.0):.4f}")
    
    print("=" * 60)
    
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
    
    # Print summary to standard output
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if experiment_type == 'synthetic':
        print(f"Best overall optimizer: {metrics.get('best_overall_optimizer', 'N/A')}")
        
        for func_name, func_metrics in metrics.items():
            if func_name != 'best_overall_optimizer':
                print(f"\n{func_name.capitalize()} function:")
                print(f"  Best optimizer: {func_metrics.get('best_optimizer', 'N/A')}")
                
                for opt_name, opt_metrics in func_metrics.items():
                    if opt_name != 'best_optimizer':
                        print(f"  {opt_name}:")
                        print(f"    Final function value: {opt_metrics.get('final_f_val', opt_metrics.get('final_value', 0.0)):.6f}")
                        print(f"    Convergence rate: {opt_metrics.get('convergence_rate', 0.0):.6f}")
    
    elif experiment_type == 'cifar10':
        print(f"Test accuracy: {metrics.get('accuracy', 0.0):.2f}%")
        print(f"Test loss: {metrics.get('test_loss', 0.0):.4f}")
        
        print("\nPer-class accuracy:")
        if 'class_accuracy' in metrics:
            for i, acc in enumerate(metrics['class_accuracy']):
                print(f"  Class {i}: {acc:.2f}%")
        else:
            print("  Class accuracy data not available")
        
        # Compare with baseline
        if 'Adam' in results and 'ACM' in results:
            acm_acc = results['ACM'].get('final_val_accuracy', 0.0)
            adam_acc = results['Adam'].get('final_val_accuracy', 0.0)
            
            print("\nComparison with baseline:")
            print(f"  ACM validation accuracy: {acm_acc:.2f}%")
            print(f"  Adam validation accuracy: {adam_acc:.2f}%")
            print(f"  Improvement: {acm_acc - adam_acc:.2f}%")
    
    elif experiment_type == 'transformer':
        print(f"Test perplexity: {metrics.get('perplexity', 0.0):.4f}")
        print(f"Test loss: {metrics.get('test_loss', 0.0):.4f}")
    
    print("\n" + "=" * 80)
    print(f"{experiment_type.upper()} experiment completed successfully!")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


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
