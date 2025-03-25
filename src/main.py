"""
Main script for running FahDiff experiments.
"""
import os
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import itertools
import random
import time
from datetime import datetime

# Import project modules
from preprocess import set_seed, prepare_dataset, generate_synthetic_graph
from train import FahDiff, train_fahdiff
from evaluate import evaluate_model, compute_graph_metrics
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.fahdiff_config import BASE_CONFIG, HYPERPARAM_GRID


def setup_experiment():
    """
    Setup experiment directories and environment.
    """
    # Create results directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def experiment_comparative_study(config, datasets, device):
    """
    Run comparative study experiment.
    
    Args:
        config: Configuration dictionary
        datasets: Dictionary of datasets with different sizes
        device: Computation device (CPU/GPU)
        
    Returns:
        DataFrame with results
    """
    print("\n" + "="*50)
    print("Experiment 1: Comparative Graph Quality and Topological Preservation")
    print("="*50)
    
    # Model variants
    model_variants = {
        "FahDiff": {"use_adaptive_force": True, "use_dynamic_schedule": True},
        "HypDiff": {"use_adaptive_force": False, "use_dynamic_schedule": False},
        "Baseline": {"use_adaptive_force": False, "use_dynamic_schedule": False, 
                     "diffusion_temperature": 0.5}
    }
    
    results = []
    
    for model_name, model_config in model_variants.items():
        print(f"\nTraining {model_name} model:")
        
        # Create model-specific config
        model_cfg = config.copy()
        model_cfg.update(model_config)
        
        for size in config["dataset_sizes"]:
            print(f"  Processing graph with {size} nodes...")
            G = datasets[size]
            
            # Train model
            model, history = train_fahdiff(model_cfg, G)
            
            # Save model
            model_path = os.path.join("models", f"{model_name}_size{size}.pt")
            torch.save(model.state_dict(), model_path)
            
            # Evaluate model
            eval_metrics = evaluate_model(model, G, model_cfg)
            
            # Record results
            results.append({
                "model": model_name,
                "nodes": size,
                "kl_div_deg": eval_metrics["kl_div_deg"],
                "avg_clustering": eval_metrics["avg_clustering"],
                "avg_path_length": eval_metrics["avg_path_length"],
                "modularity": eval_metrics["modularity"],
                "final_loss": history["loss"][-1]
            })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join("logs", "comparative_study_results.csv")
    df_results.to_csv(results_path, index=False)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for model_name in model_variants.keys():
        df_model = df_results[df_results["model"] == model_name]
        plt.plot(df_model["nodes"], df_model["kl_div_deg"], marker="o", label=model_name)
    
    plt.xlabel("Number of Nodes")
    plt.ylabel("KL-divergence (Degree Distribution)")
    plt.title("Comparative Degree Distribution KL-divergence")
    plt.legend()
    plt.savefig(os.path.join("logs", "comparative_degree_kl_divergence.pdf"), 
               format="pdf", bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved comparative study results to {results_path}")
    return df_results


def experiment_ablation_study(config, device):
    """
    Run ablation study experiment.
    
    Args:
        config: Configuration dictionary
        device: Computation device (CPU/GPU)
        
    Returns:
        DataFrame with results
    """
    print("\n" + "="*50)
    print("Experiment 2: Ablation Study")
    print("="*50)
    
    # Fixed graph size for ablation study
    # Use first size if only one size is available
    if len(config["dataset_sizes"]) > 1:
        n_nodes = config["dataset_sizes"][1]  # Use middle size
    else:
        n_nodes = config["dataset_sizes"][0]  # Use first size
    G = generate_synthetic_graph(n_nodes=n_nodes, seed=config["seed"])
    
    # Model variants for ablation
    ablation_variants = {
        "Full FahDiff": {"use_adaptive_force": True, "use_dynamic_schedule": True},
        "No Adaptive Force": {"use_adaptive_force": False, "use_dynamic_schedule": True},
        "Static Schedule": {"use_adaptive_force": True, "use_dynamic_schedule": False}
    }
    
    results = []
    histories = {}
    
    for variant_name, variant_config in ablation_variants.items():
        print(f"\nTraining {variant_name}:")
        
        # Create variant-specific config
        variant_cfg = config.copy()
        variant_cfg.update(variant_config)
        
        # Train model
        model, history = train_fahdiff(variant_cfg, G)
        
        # Save model
        model_path = os.path.join("models", f"ablation_{variant_name.replace(' ', '_')}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Save history for plotting
        histories[variant_name] = history
        
        # Evaluate model
        eval_metrics = evaluate_model(model, G, variant_cfg)
        
        # Record results
        results.append({
            "variant": variant_name,
            "kl_div_deg": eval_metrics["kl_div_deg"],
            "avg_clustering": eval_metrics["avg_clustering"],
            "avg_path_length": eval_metrics["avg_path_length"],
            "modularity": eval_metrics["modularity"],
            "final_loss": history["loss"][-1]
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join("logs", "ablation_study_results.csv")
    df_results.to_csv(results_path, index=False)
    
    # Plot training loss curves
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(next(iter(histories.values()))["loss"]) + 1)
    
    for variant_name, history in histories.items():
        plt.plot(epochs, history["loss"], label=variant_name, marker="o" if variant_name == "Full FahDiff" else None)
    
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Ablation Study: Training Loss Comparison")
    plt.legend()
    plt.savefig(os.path.join("logs", "ablation_study_loss.pdf"), 
               format="pdf", bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved ablation study results to {results_path}")
    return df_results


def experiment_hyperparameter_sensitivity(config, device):
    """
    Run hyperparameter sensitivity experiment.
    
    Args:
        config: Configuration dictionary
        device: Computation device (CPU/GPU)
        
    Returns:
        DataFrame with results
    """
    print("\n" + "="*50)
    print("Experiment 3: Hyperparameter Sensitivity and Stability Analysis")
    print("="*50)
    
    # Fixed graph size for hyperparameter study
    if len(config["dataset_sizes"]) > 0:
        n_nodes = config["dataset_sizes"][0]  # Use smallest size for faster runs
    else:
        n_nodes = 10  # Default to 10 nodes if no sizes are specified
    G = generate_synthetic_graph(n_nodes=n_nodes, seed=config["seed"])
    
    # Create list of hyperparameter combinations
    # For test runs, use a smaller grid
    if config.get("test_run", False):
        # Use just one combination for testing
        hyperparameter_combinations = [{
            "diffusion_temperature": HYPERPARAM_GRID["diffusion_temperature"][0],
            "curvature_param": HYPERPARAM_GRID["curvature_param"][0],
            "force_learning_rate": HYPERPARAM_GRID["force_learning_rate"][0]
        }]
    else:
        # Use full grid for regular runs
        keys, values = zip(*HYPERPARAM_GRID.items())
        hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    for hp_idx, hp in enumerate(hyperparameter_combinations):
        print(f"\nHyperparameter combination {hp_idx+1}/{len(hyperparameter_combinations)}:")
        print(f"  diffusion_temperature: {hp['diffusion_temperature']}")
        print(f"  curvature_param: {hp['curvature_param']}")
        print(f"  force_learning_rate: {hp['force_learning_rate']}")
        
        # Create hyperparameter-specific config
        hp_cfg = config.copy()
        hp_cfg.update(hp)
        
        # Train model
        model, history = train_fahdiff(hp_cfg, G)
        
        # Save model
        model_name = f"hp_temp{hp['diffusion_temperature']}_curv{hp['curvature_param']}_flr{hp['force_learning_rate']}"
        model_path = os.path.join("models", f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Evaluate model
        eval_metrics = evaluate_model(model, G, hp_cfg)
        
        # Find convergence epoch (minimum loss)
        convergence_epoch = np.argmin(np.array(history["loss"])) + 1
        
        # Record results
        results.append({
            "diffusion_temperature": hp["diffusion_temperature"],
            "curvature_param": hp["curvature_param"],
            "force_learning_rate": hp["force_learning_rate"],
            "kl_div_deg": eval_metrics["kl_div_deg"],
            "avg_clustering": eval_metrics["avg_clustering"],
            "avg_path_length": eval_metrics["avg_path_length"],
            "modularity": eval_metrics["modularity"],
            "final_loss": history["loss"][-1],
            "convergence_epoch": convergence_epoch
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join("logs", "hyperparameter_sensitivity_results.csv")
    df_results.to_csv(results_path, index=False)
    
    # Plot heatmap for a fixed force_learning_rate
    # Skip heatmap for test runs with only one combination
    if config.get("test_run", False) or len(hyperparameter_combinations) <= 1:
        print("\nSkipping heatmap generation for test run with limited hyperparameters")
        return df_results
        
    fixed_lr = HYPERPARAM_GRID["force_learning_rate"][1]  # Middle value
    df_fixed = df_results[df_results["force_learning_rate"] == fixed_lr]
    
    # Create pivot table for heatmap
    pivot_loss = df_fixed.pivot(
        index="diffusion_temperature", 
        columns="curvature_param", 
        values="final_loss"
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_loss, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Final Loss across Diffusion Temperature and Curvature Parameter\n(force_learning_rate = {fixed_lr})")
    plt.xlabel("Curvature Parameter")
    plt.ylabel("Diffusion Temperature")
    plt.savefig(os.path.join("logs", "hyperparameter_sensitivity_heatmap.pdf"), 
               format="pdf", bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved hyperparameter sensitivity results to {results_path}")
    return df_results


def run_test_experiment():
    """
    Run a quick test of all experiments with minimal settings.
    """
    print("\n" + "="*50)
    print("Running Test Experiments")
    print("="*50)
    
    # Create test config with minimal settings
    test_config = BASE_CONFIG.copy()
    test_config.update({
        "epochs": 3,
        "dataset_sizes": [10],
        "test_run": True
    })
    
    # Setup experiment
    device = setup_experiment()
    
    # Prepare datasets
    datasets = prepare_dataset(test_config)
    
    # Run experiments
    experiment_comparative_study(test_config, datasets, device)
    experiment_ablation_study(test_config, device)
    experiment_hyperparameter_sensitivity(test_config, device)
    
    print("\nTest experiments completed successfully!")


def main():
    """
    Main function to run all experiments.
    """
    print("="*50)
    print("Force-Enhanced Adaptive Hyperbolic Diffusion (FahDiff) Experiments")
    print("="*50)
    
    # Set random seed for reproducibility
    set_seed(BASE_CONFIG["seed"])
    
    # Setup experiment
    device = setup_experiment()
    
    # Check if this is a test run
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_experiment()
        return
    
    # Prepare datasets
    print("\nPreparing datasets...")
    datasets = prepare_dataset(BASE_CONFIG)
    
    # Run experiments
    experiment_comparative_study(BASE_CONFIG, datasets, device)
    experiment_ablation_study(BASE_CONFIG, device)
    experiment_hyperparameter_sensitivity(BASE_CONFIG, device)
    
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    import sys
    main()
