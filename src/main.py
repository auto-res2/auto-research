"""
Main script for running STEM experiments.
Orchestrates the entire process from data preprocessing to evaluation.
"""

import torch
import os
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
    RUN_CONTROLLED_SYNTHETIC, RUN_ABLATION_STUDY, RUN_DOMAIN_SHIFT
)

torch.manual_seed(42)

def experiment_controlled_synthetic():
    """
    Experiment 1: Controlled Synthetic Dataset Experiment
    Compares STEM model with a baseline additive model.
    """
    print("\n" + "="*80)
    print("Experiment 1: Controlled Synthetic Dataset Experiment")
    print("="*80)
    
    data = generate_synthetic_data(num_samples=SYNTHETIC_SAMPLES)
    
    print("Initializing models...")
    stem_model = STEMModel(hidden_dim=MODEL_HIDDEN_DIM)
    baseline_model = AdditiveModel(hidden_dim=MODEL_HIDDEN_DIM)
    
    print("Training STEM model...")
    stem_loss_history = train_model(stem_model, data, num_epochs=TRAIN_EPOCHS, lr=LEARNING_RATE)
    print("Training Baseline Additive model...")
    baseline_loss_history = train_model(baseline_model, data, num_epochs=TRAIN_EPOCHS, lr=LEARNING_RATE)
    
    plot_training_curve(stem_loss_history, figure_topic="training_loss", condition="STEM", pair=1)
    plot_training_curve(baseline_loss_history, figure_topic="training_loss", condition="baseline", pair=2)

    print("Evaluating STEM model:")
    evaluate_model(stem_model, data)
    print("Evaluating Baseline model:")
    evaluate_model(baseline_model, data)

def experiment_ablation_study():
    """
    Experiment 2: Ablation Study on Subtractive Mechanism and Auxiliary Metadata
    Compares different model variants.
    """
    print("\n" + "="*80)
    print("Experiment 2: Ablation Study on Subtractive Mechanism and Auxiliary Metadata")
    print("="*80)
    
    data = generate_metadata_data(num_samples=METADATA_SAMPLES, metadata_dim=METADATA_DIM)
    
    print("Initializing models...")
    additive_model = AdditiveModel(hidden_dim=MODEL_HIDDEN_DIM)
    additive_meta_model = AdditiveWithMetadataModel(hidden_dim=MODEL_HIDDEN_DIM, metadata_dim=METADATA_DIM)
    stem_meta_model = STEMWithMetadataModel(hidden_dim=MODEL_HIDDEN_DIM, metadata_dim=METADATA_DIM)
    
    print("Training Pure Additive Model (Variant 1)...")
    loss_additive = train_model(additive_model, data, num_epochs=TRAIN_EPOCHS, lr=LEARNING_RATE)
    print("Training Additive with Metadata Model (Variant 2)...")
    loss_additive_meta = train_model(additive_meta_model, data, num_epochs=TRAIN_EPOCHS, use_metadata=True, lr=LEARNING_RATE)
    print("Training Full STEM with Metadata Model (Variant 3)...")
    loss_stem_meta = train_model(stem_meta_model, data, num_epochs=TRAIN_EPOCHS, use_metadata=True, lr=LEARNING_RATE)
    
    plot_training_curve(loss_additive, figure_topic="training_loss", condition="additive", pair=1)
    plot_training_curve(loss_additive_meta, figure_topic="training_loss", condition="additive_meta", pair=2)
    plot_training_curve(loss_stem_meta, figure_topic="training_loss", condition="STEM_meta", pair=3)
    
    print("Evaluating Variant 1 (Pure Additive):")
    evaluate_model(additive_model, data)
    print("Evaluating Variant 2 (Additive with Metadata):")
    evaluate_model(additive_meta_model, data)
    print("Evaluating Variant 3 (Full STEM with Metadata):")
    evaluate_model(stem_meta_model, data)

def experiment_domain_shift():
    """
    Experiment 3: Robustness Under Domain Shifts
    Tests STEM model's performance when trained on one domain and tested on another.
    """
    print("\n" + "="*80)
    print("Experiment 3: Robustness Under Domain Shifts")
    print("="*80)
    
    data = generate_domain_data(num_samples=DOMAIN_SAMPLES)
    train_data = [sample for sample in data if sample[-1] != "academic"]
    test_data = [sample for sample in data if sample[-1] == "academic"]
    print(f"Training samples: {len(train_data)} | Testing samples: {len(test_data)}")
    
    domain_model = STEMModel(hidden_dim=MODEL_HIDDEN_DIM)
    print("Training STEM model on non-academic domain...")
    train_model(domain_model, train_data, num_epochs=TRAIN_EPOCHS, lr=LEARNING_RATE)
    
    print("Evaluating model on academic domain (domain shift)...")
    evaluate_model(domain_model, test_data)
    
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
    
    plot_domain_shift_results(targets, predictions)

def main():
    """Main function to run all experiments."""
    print("Starting STEM experiments...")
    
    prepare_logs_directory()
    
    initialize_models()
    
    if RUN_CONTROLLED_SYNTHETIC:
        experiment_controlled_synthetic()
    
    if RUN_ABLATION_STUDY:
        experiment_ablation_study()
    
    if RUN_DOMAIN_SHIFT:
        experiment_domain_shift()
    
    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
