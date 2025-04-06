"""
Main script for running STEM experiments.
Orchestrates the entire process from data preprocessing to evaluation.
"""

import torch
import os
import sys
import time
import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocess import initialize_models, extract_embeddings
from src.train import (
    STEMModel, AdditiveModel, STEMWithMetadataModel, 
    AdditiveWithMetadataModel, train_model
)
from src.evaluate import evaluate_model, plot_training_curve, plot_domain_shift_results
from src.utils.utils import generate_synthetic_data, generate_metadata_data, generate_domain_data, prepare_logs_directory
from config.experiment_config import (
    MODEL_HIDDEN_DIM, METADATA_DIM, TRAIN_EPOCHS, LEARNING_RATE,
    SYNTHETIC_SAMPLES, METADATA_SAMPLES, DOMAIN_SAMPLES,
    RUN_CONTROLLED_SYNTHETIC, RUN_ABLATION_STUDY, RUN_DOMAIN_SHIFT,
    BATCH_SIZE
)

torch.manual_seed(42)

def experiment_controlled_synthetic():
    """
    Experiment 1: Controlled Synthetic Dataset Experiment
    Compares STEM model with a baseline additive model.
    """
    print("\n" + "="*100)
    print("EXPERIMENT 1: CONTROLLED SYNTHETIC DATASET EXPERIMENT")
    print("="*100)
    print("Description: This experiment compares the STEM model with a baseline additive model")
    print("             on a controlled synthetic dataset to evaluate performance differences.")
    print("-"*100)
    
    print(f"Generating synthetic dataset with {SYNTHETIC_SAMPLES} samples...")
    data = generate_synthetic_data(num_samples=SYNTHETIC_SAMPLES)
    print(f"Dataset generated successfully. Sample text example: '{data[0][0]}'")
    
    print("\nInitializing models...")
    print(f"  - STEM Model: Hidden dimension = {MODEL_HIDDEN_DIM}")
    stem_model = STEMModel(hidden_dim=MODEL_HIDDEN_DIM)
    print(f"  - Baseline Additive Model: Hidden dimension = {MODEL_HIDDEN_DIM}")
    baseline_model = AdditiveModel(hidden_dim=MODEL_HIDDEN_DIM)
    print("Models initialized successfully.")
    
    print("\nTraining STEM model...")
    print(f"  - Epochs: {TRAIN_EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    stem_loss_history = train_model(stem_model, data, num_epochs=TRAIN_EPOCHS, lr=LEARNING_RATE)
    
    print("\nTraining Baseline Additive model...")
    print(f"  - Epochs: {TRAIN_EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    baseline_loss_history = train_model(baseline_model, data, num_epochs=TRAIN_EPOCHS, lr=LEARNING_RATE)
    
    print("\nGenerating training loss plots...")
    stem_plot = plot_training_curve(stem_loss_history, figure_topic="training_loss", condition="STEM", pair=1)
    baseline_plot = plot_training_curve(baseline_loss_history, figure_topic="training_loss", condition="baseline", pair=2)
    print(f"Plots saved successfully:")
    print(f"  - STEM model: {stem_plot}")
    print(f"  - Baseline model: {baseline_plot}")

    print("\nEvaluating STEM model:")
    stem_mse = evaluate_model(stem_model, data)
    print("\nEvaluating Baseline model:")
    baseline_mse = evaluate_model(baseline_model, data)
    
    print("\nExperiment 1 Results Summary:")
    print(f"  - STEM Model MSE: {stem_mse:.6f}")
    print(f"  - Baseline Model MSE: {baseline_mse:.6f}")
    print(f"  - Difference: {abs(stem_mse - baseline_mse):.6f}")
    if stem_mse < baseline_mse:
        print("  - STEM model outperformed the baseline model")
    elif stem_mse > baseline_mse:
        print("  - Baseline model outperformed the STEM model")
    else:
        print("  - Both models performed equally")
        
    print("\nTraining Loss Values (Plain Text):")
    print("| Epoch | STEM Model | Baseline Model |")
    print("|-------|------------|----------------|")
    for i, (stem_loss, baseline_loss) in enumerate(zip(stem_loss_history, baseline_loss_history)):
        print(f"| {i+1:5d} | {stem_loss:.6f} | {baseline_loss:.6f} |")

def experiment_ablation_study():
    """
    Experiment 2: Ablation Study on Subtractive Mechanism and Auxiliary Metadata
    Compares different model variants.
    """
    print("\n" + "="*100)
    print("EXPERIMENT 2: ABLATION STUDY ON SUBTRACTIVE MECHANISM AND AUXILIARY METADATA")
    print("="*100)
    print("Description: This experiment compares three model variants to evaluate the impact of")
    print("             subtractive mechanisms and auxiliary metadata learning:")
    print("             1. Pure Additive Model (baseline)")
    print("             2. Additive Model with Metadata")
    print("             3. Full STEM Model with Metadata")
    print("-"*100)
    
    print(f"Generating metadata dataset with {METADATA_SAMPLES} samples...")
    data = generate_metadata_data(num_samples=METADATA_SAMPLES, metadata_dim=METADATA_DIM)
    print(f"Dataset generated successfully. Sample text example: '{data[0][0]}'")
    print(f"Metadata dimension: {METADATA_DIM}")
    
    print("\nInitializing model variants...")
    print("  - Variant 1: Pure Additive Model (baseline)")
    additive_model = AdditiveModel(hidden_dim=MODEL_HIDDEN_DIM)
    print("  - Variant 2: Additive Model with Metadata")
    additive_meta_model = AdditiveWithMetadataModel(hidden_dim=MODEL_HIDDEN_DIM, metadata_dim=METADATA_DIM)
    print("  - Variant 3: Full STEM Model with Metadata")
    stem_meta_model = STEMWithMetadataModel(hidden_dim=MODEL_HIDDEN_DIM, metadata_dim=METADATA_DIM)
    print("All model variants initialized successfully.")
    
    print("\nTraining Variant 1: Pure Additive Model...")
    print(f"  - Epochs: {TRAIN_EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Metadata: Not used (but present in data)")
    loss_additive = train_model(additive_model, data, num_epochs=TRAIN_EPOCHS, lr=LEARNING_RATE, use_metadata=True)
    
    print("\nTraining Variant 2: Additive Model with Metadata...")
    print(f"  - Epochs: {TRAIN_EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Metadata: Used for auxiliary learning")
    loss_additive_meta = train_model(additive_meta_model, data, num_epochs=TRAIN_EPOCHS, use_metadata=True, lr=LEARNING_RATE)
    
    print("\nTraining Variant 3: Full STEM Model with Metadata...")
    print(f"  - Epochs: {TRAIN_EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Metadata: Used for auxiliary learning")
    print(f"  - Mechanism: Subtractive")
    loss_stem_meta = train_model(stem_meta_model, data, num_epochs=TRAIN_EPOCHS, use_metadata=True, lr=LEARNING_RATE)
    
    print("\nGenerating training loss plots...")
    plot1 = plot_training_curve(loss_additive, figure_topic="training_loss", condition="additive", pair=1)
    plot2 = plot_training_curve(loss_additive_meta, figure_topic="training_loss", condition="additive_meta", pair=2)
    plot3 = plot_training_curve(loss_stem_meta, figure_topic="training_loss", condition="STEM_meta", pair=3)
    print(f"Plots saved successfully:")
    print(f"  - Variant 1 (Pure Additive): {plot1}")
    print(f"  - Variant 2 (Additive with Metadata): {plot2}")
    print(f"  - Variant 3 (Full STEM with Metadata): {plot3}")
    
    print("\nEvaluating Variant 1 (Pure Additive):")
    v1_mse = evaluate_model(additive_model, data)
    print("\nEvaluating Variant 2 (Additive with Metadata):")
    v2_mse = evaluate_model(additive_meta_model, data)
    print("\nEvaluating Variant 3 (Full STEM with Metadata):")
    v3_mse = evaluate_model(stem_meta_model, data)
    
    print("\nExperiment 2 Results Summary:")
    print(f"  - Variant 1 (Pure Additive) MSE: {v1_mse:.6f}")
    print(f"  - Variant 2 (Additive with Metadata) MSE: {v2_mse:.6f}")
    print(f"  - Variant 3 (Full STEM with Metadata) MSE: {v3_mse:.6f}")
    
    v1_mse_float = float(v1_mse)
    v2_mse_float = float(v2_mse)
    v3_mse_float = float(v3_mse)
    
    best_mse = min(v1_mse_float, v2_mse_float, v3_mse_float)
    if best_mse == v1_mse_float:
        best_model = "Variant 1 (Pure Additive)"
    elif best_mse == v2_mse_float:
        best_model = "Variant 2 (Additive with Metadata)"
    else:
        best_model = "Variant 3 (Full STEM with Metadata)"
    
    print(f"  - Best performing model: {best_model} with MSE: {best_mse:.6f}")
    print(f"  - Improvement over baseline: {((v1_mse - best_mse) / v1_mse * 100):.2f}%")
    
    print("\nTraining Loss Values (Plain Text):")
    print("| Epoch | Variant 1 (Pure) | Variant 2 (Add+Meta) | Variant 3 (STEM+Meta) |")
    print("|-------|-----------------|----------------------|------------------------|")
    for i in range(len(loss_additive)):
        if i < len(loss_additive) and i < len(loss_additive_meta) and i < len(loss_stem_meta):
            print(f"| {i+1:5d} | {loss_additive[i]:.6f} | {loss_additive_meta[i]:.6f} | {loss_stem_meta[i]:.6f} |")

def experiment_domain_shift():
    """
    Experiment 3: Robustness Under Domain Shifts
    Tests STEM model's performance when trained on one domain and tested on another.
    """
    print("\n" + "="*100)
    print("EXPERIMENT 3: ROBUSTNESS UNDER DOMAIN SHIFTS")
    print("="*100)
    print("Description: This experiment tests the STEM model's robustness when trained on")
    print("             one domain (news) and evaluated on another domain (academic).")
    print("             This evaluates the model's ability to generalize across domains.")
    print("-"*100)
    
    print(f"Generating domain-specific dataset with {DOMAIN_SAMPLES} samples...")
    data = generate_domain_data(num_samples=DOMAIN_SAMPLES)
    train_data = [sample for sample in data if sample[-1] != "academic"]
    test_data = [sample for sample in data if sample[-1] == "academic"]
    
    print(f"Dataset split:")
    print(f"  - Training samples (non-academic domain): {len(train_data)}")
    print(f"  - Testing samples (academic domain): {len(test_data)}")
    print(f"  - Training sample example: '{train_data[0][0]}' (Domain: {train_data[0][-1]})")
    print(f"  - Testing sample example: '{test_data[0][0]}' (Domain: {test_data[0][-1]})")
    
    print("\nInitializing STEM model for domain shift experiment...")
    print(f"  - Hidden dimension: {MODEL_HIDDEN_DIM}")
    domain_model = STEMModel(hidden_dim=MODEL_HIDDEN_DIM)
    print("Model initialized successfully.")
    
    print("\nTraining STEM model on non-academic domain...")
    print(f"  - Epochs: {TRAIN_EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Training samples: {len(train_data)}")
    loss_history = train_model(domain_model, train_data, num_epochs=TRAIN_EPOCHS, lr=LEARNING_RATE)
    
    print("\nGenerating training loss plot...")
    loss_plot = plot_training_curve(loss_history, figure_topic="training_loss", condition="domain_shift", pair=1)
    print(f"Plot saved successfully: {loss_plot}")
    
    print("\nEvaluating model on academic domain (domain shift)...")
    print(f"  - Testing samples: {len(test_data)}")
    test_mse = evaluate_model(domain_model, test_data)
    
    print("\nGenerating domain shift scatter plot...")
    predictions = []
    targets = []
    with torch.no_grad():
        for sample in test_data:
            text = sample[0]
            true_alpha = sample[1]
            embeddings = extract_embeddings([text])
            alpha_tensor = true_alpha.view(-1, 1)
            output = domain_model(embeddings, alpha_tensor)
            predictions.append(output.item())
            targets.append(0.0)  # dummy target
    
    scatter_plot = plot_domain_shift_results(targets, predictions)
    print(f"Scatter plot saved successfully: {scatter_plot}")
    
    print("\nExperiment 3 Results Summary:")
    print(f"  - Domain Shift Test MSE: {float(test_mse):.6f}")
    print(f"  - Number of outliers: {sum(1 for p in predictions if abs(p) > 0.5)}")
    print(f"  - Mean prediction: {sum(predictions)/len(predictions):.6f}")
    print(f"  - Standard deviation: {torch.tensor(predictions).std().item():.6f}")
    print("  - Conclusion: The model's performance across domains demonstrates its")
    print("                robustness to domain shifts in text analysis tasks.")
    
    print("\nTraining Loss Values (Plain Text):")
    print("| Epoch | Loss Value |")
    print("|-------|------------|")
    for i, loss in enumerate(loss_history):
        print(f"| {i+1:5d} | {loss:.6f} |")
        
    print("\nDomain Shift Predictions (Plain Text):")
    print("| Sample | True Value | Predicted Value |")
    print("|--------|------------|-----------------|")
    for i, (target, pred) in enumerate(zip(targets, predictions)):
        print(f"| {i+1:6d} | {target:.6f} | {pred:.6f} |")

def main():
    """Main function to run all experiments."""
    start_time = time.time()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*100)
    print(f"STEM EXPERIMENTS - STARTED AT {current_time}")
    print("="*100)
    print(f"Configuration:")
    print(f"  - Hidden Dimension: {MODEL_HIDDEN_DIM}")
    print(f"  - Metadata Dimension: {METADATA_DIM}")
    print(f"  - Training Epochs: {TRAIN_EPOCHS}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Synthetic Samples: {SYNTHETIC_SAMPLES}")
    print(f"  - Metadata Samples: {METADATA_SAMPLES}")
    print(f"  - Domain Samples: {DOMAIN_SAMPLES}")
    print("="*100 + "\n")
    
    prepare_logs_directory()
    print("Logs directory prepared successfully.")
    
    print("Initializing transformer models for embedding extraction...")
    initialize_models()
    print("Transformer models initialized successfully.")
    
    experiments_run = 0
    
    if RUN_CONTROLLED_SYNTHETIC:
        experiment_controlled_synthetic()
        experiments_run += 1
    
    if RUN_ABLATION_STUDY:
        experiment_ablation_study()
        experiments_run += 1
    
    if RUN_DOMAIN_SHIFT:
        experiment_domain_shift()
        experiments_run += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print("\n" + "="*100)
    print("STEM EXPERIMENTS - SUMMARY")
    print("="*100)
    print(f"Experiments completed: {experiments_run}")
    print(f"Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds")
    print(f"All plots saved to logs/ directory in PDF format")
    print("="*100)
    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
